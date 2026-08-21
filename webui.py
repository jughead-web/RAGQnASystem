import os
import streamlit as st
import ner_model as ner
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re
import json
import logging

from typing import Dict, List, Tuple

from config import settings
from logging_setup import setup_logging
from kg_client import (
    KGClient,
    build_attribute_prompt,
    build_relation_prompt,
)
from intent_router import execute_intents_with_evidence

from evidence import (
    Evidence,
    build_grounded_prompt,
)

from context_resolver import (
    is_source_query,
    get_last_assistant_turn,
    format_evidence_answer,
    resolve_followup_query,
)

from kg_answer_formatter import (
    is_pure_kg_evidence,
    format_kg_answer,
)

from vector_rag import VectorRetriever
from medical_agent import MedicalAgent

setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def load_vector_retriever():

    try:
        return VectorRetriever()

    except RuntimeError as e:

        if "already accessed" in str(e):

            st.warning(
                "Vector RAG 已由 Medical Agent 加载，跳过重复初始化"
            )

            return None

        raise e


@st.cache_resource
def load_medical_agent():
    """
    加载 Multi-Memory Medical Agent
    """
    return MedicalAgent(
        model=settings.OLLAMA_QWEN_MODEL
    )


# ======================================================================
# Retrieval Router V2
# ======================================================================

VECTOR_FORCE_HINTS = (
    "控制目标",
    "血压目标",
    "目标血压",
    "目标值",
    "达标值",
    "达标标准",
    "控制标准",
    "指南怎么说",
    "指南推荐",
    "指南建议",
    "推荐标准",
    "推荐目标",
    "随访",
    "复查",
    "多久复查",
    "多久随访",
    "监测频率",
    "随访频率",
    "治疗后",
    "开始药物治疗后",
)

HYBRID_HINTS = (
    "为什么",
    "为什么需要",
    "原理",
    "机制",
    "长期",
    "生活方式",
    "如何改善",
    "怎么改善",
    "注意事项",
    "有什么影响",
    "如何理解",
    "解释一下",
    "怎么回事",
    "如何管理",
    "怎么管理",
)


def decide_retrieval_mode_v2(
    query: str,
    kg_evidence: List[Evidence],
) -> Tuple[str, str]:
    """
    返回:
        (retrieval_mode, reason)

    retrieval_mode:
        KG
        VECTOR
        HYBRID
    """
    text = (query or "").strip()

    force_hits = [
        hint
        for hint in VECTOR_FORCE_HINTS
        if hint in text
    ]

    if force_hits:
        return (
            "VECTOR",
            "命中规范/目标类关键词: "
            + "、".join(force_hits),
        )

    hybrid_hits = [
        hint
        for hint in HYBRID_HINTS
        if hint in text
    ]

    if hybrid_hits:
        if kg_evidence:
            return (
                "HYBRID",
                "命中解释型关键词且存在 KG 证据: "
                + "、".join(hybrid_hits),
            )

        return (
            "VECTOR",
            "命中解释型关键词且 KG 无直接证据: "
            + "、".join(hybrid_hits),
        )

    if kg_evidence:
        return (
            "KG",
            "存在结构化 KG 证据，按事实型查询处理",
        )

    return (
        "VECTOR",
        "KG 未命中有效证据，回退到文档向量检索",
    )



@st.cache_resource
def load_model(cache_model: str):
    """加载 NER 推理所需的全部资源（被 streamlit 缓存）。

    返回 ``(glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device)``，
    其中 ``glm_*`` 已废弃返回 ``None``，仅为保持原签名不破坏调用方。
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # ChatGLM 路径已废弃；保留 None 占位避免下游解包错误
    glm_model = None
    glm_tokenizer = None
    # 加载 Bert 模型
    with open(os.path.join(settings.TMP_DIR, 'tag2idx.npy'), 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = ner.rule_find()
    tfidf_r = ner.tfidf_alignment()
    model_name = settings.NER_MODEL_NAME
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = ner.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(
        os.path.join(settings.MODEL_DIR, f'{cache_model}.pt'),
        map_location=device
    ))
    bert_model = bert_model.to(device)
    bert_model.eval()
    return glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device



def Intent_Recognition(
    query: str,
    choice: str,
) -> str:
    """
    稳定版意图识别：规则优先，LLM 只做兜底。

    这样可以保证：
    - “高血压的症状”这类明确问题不再依赖 7B 模型随机分类；
    - 开放式解释问题不强行塞进 KG 意图；
    - 多意图问题仍可识别；
    - 返回值保持为 JSON 数组字符串，兼容 intent_router.route_intents().
    """

    allowed_intents = [
        "查询疾病简介",
        "查询疾病病因",
        "查询疾病预防措施",
        "查询疾病治疗周期",
        "查询治愈概率",
        "查询疾病易感人群",
        "查询疾病所需药品",
        "查询疾病宜吃食物",
        "查询疾病忌吃食物",
        "查询疾病所需检查项目",
        "查询疾病所属科目",
        "查询疾病的症状",
        "查询疾病的治疗方法",
        "查询疾病的并发疾病",
        "查询药品的生产商",
    ]

    q = query.strip()
    matched: List[str] = []

    def add(intent: str) -> None:
        if intent not in matched:
            matched.append(intent)

    # --------------------------------------------------------------
    # A. 明确、可确定的 KG 意图：规则优先
    # --------------------------------------------------------------
    rule_groups = [
        (
            ["症状", "什么表现", "有哪些表现", "临床表现"],
            "查询疾病的症状",
        ),
        (
            ["怎么治疗", "如何治疗", "怎么治", "治疗方法", "如何医治"],
            "查询疾病的治疗方法",
        ),
        (
            ["吃什么药", "用什么药", "使用什么药", "有哪些药", "什么药", "药品有哪些", "需要哪些药"],
            "查询疾病所需药品",
        ),
        (
            ["做什么检查", "需要什么检查", "检查项目", "需要检查", "怎么检查"],
            "查询疾病所需检查项目",
        ),
        (
            ["怎么预防", "如何预防", "预防措施"],
            "查询疾病预防措施",
        ),
        (
            ["治疗周期", "多久能治", "治疗多久"],
            "查询疾病治疗周期",
        ),
        (
            ["治愈概率", "能治好吗", "治愈率", "能不能治好"],
            "查询治愈概率",
        ),
        (
            ["易感人群", "哪些人容易得", "什么人容易得"],
            "查询疾病易感人群",
        ),
        (
            ["宜吃", "适合吃什么", "吃什么好"],
            "查询疾病宜吃食物",
        ),
        (
            ["忌吃", "不能吃什么", "不宜吃什么"],
            "查询疾病忌吃食物",
        ),
        (
            ["属于什么科", "挂什么科", "所属科目", "哪个科"],
            "查询疾病所属科目",
        ),
        (
            ["并发症", "并发疾病", "会引起什么病"],
            "查询疾病的并发疾病",
        ),
        (
            ["生产商", "哪个公司生产", "哪家生产"],
            "查询药品的生产商",
        ),
        (
            ["疾病简介", "介绍一下", "是什么病", "什么是"],
            "查询疾病简介",
        ),
    ]

    for keywords, intent in rule_groups:
        if any(keyword in q for keyword in keywords):
            add(intent)

    # “病因/原因”必须更谨慎，避免把
    # “为什么需要长期生活方式干预”误判成疾病病因。
    cause_patterns = [
        "病因",
        "什么原因引起",
        "什么原因导致",
        "为什么会得",
        "为什么会患",
        "怎么引起的",
        "如何引起",
        "发病原因",
    ]
    if any(pattern in q for pattern in cause_patterns):
        add("查询疾病病因")

    # 规则已经明确识别到意图时，直接返回，不调用 LLM。
    if matched:
        result = json.dumps(
            matched[:3],
            ensure_ascii=False,
        )
        logger.debug(
            "规则意图识别结果: %s",
            result,
        )
        return result

    # --------------------------------------------------------------
    # B. 规则无法确定时，才让 LLM 做兜底分类
    # --------------------------------------------------------------
    prompt = f"""
你是一个医疗知识图谱查询意图分类器。

你只做分类，不回答医学问题。

允许的意图：
{json.dumps(allowed_intents, ensure_ascii=False)}

要求：
1. 只选择用户明确询问、且能够由上述知识图谱意图直接表达的类别。
2. 不要因为出现疾病名称就自动增加“查询疾病简介”。
3. 不要扩展用户没有询问的内容。
4. 如果问题属于开放式解释、生活方式原因、个体化健康咨询，
   而这些类别无法准确表达，则返回空列表。
5. 最多返回 3 个意图。
6. 只输出 JSON 对象，不要输出解释。

用户问题：
{query}

输出格式：
{{"intents": []}}
""".strip()

    try:
        result = ollama.generate(
            model=choice,
            prompt=prompt,
            format="json",
            options={
                "temperature": 0,
                "seed": 42,
            },
        )

        raw = result["response"]
        data = json.loads(raw)
        intents = data.get("intents", [])

        if not isinstance(intents, list):
            intents = []

        valid_intents: List[str] = []
        for intent in intents:
            if (
                intent in allowed_intents
                and intent not in valid_intents
            ):
                valid_intents.append(intent)

        rec_result = json.dumps(
            valid_intents[:3],
            ensure_ascii=False,
        )

    except Exception as exc:
        logger.exception(
            "结构化意图识别失败: %s",
            exc,
        )
        rec_result = "[]"

    logger.debug(
        "LLM兜底意图识别结果: %s",
        rec_result,
    )
    return rec_result

def add_shuxing_prompt(entity, shuxing, client):
    """[转发] 查询疾病属性并生成 ``<提示>...</提示>`` 文本。

    历史接口保留：第三个参数 ``client`` 既可以是 ``py2neo.Graph``，也可以是
    :class:`kg_client.KGClient`。统一委托给 :class:`KGClient` 实现。
    """
    kg = client if isinstance(client, KGClient) else KGClient(client)
    value = kg.get_disease_attribute(entity, shuxing)
    return build_attribute_prompt(entity, shuxing, value)


def add_lianxi_prompt(entity, lianxi, target, client):
    """[转发] 查询疾病关系并生成 ``<提示>...</提示>`` 文本。"""
    kg = client if isinstance(client, KGClient) else KGClient(client)
    items = kg.get_related_entities(entity, lianxi, target)
    return build_relation_prompt(entity, lianxi, items)
def generate_prompt(
    response: str,
    query: str,
    client,
    bert_model,
    bert_tokenizer,
    rule,
    tfidf_r,
    device,
    idx2tag,
) -> Tuple[
    str,
    str,
    Dict[str, str],
    List[Evidence],
]:
    """
    新版 RAG Prompt 生成。

    返回：
        prompt
        intents_str
        entities
        evidence_list
    """

    # --------------------------------------------------------------
    # 1. NER
    # --------------------------------------------------------------

    entities = ner.get_ner_result(
        bert_model,
        bert_tokenizer,
        query,
        rule,
        tfidf_r,
        device,
        idx2tag,
    )

    kg = (
        client
        if isinstance(client, KGClient)
        else KGClient(client)
    )

    evidence_list: List[Evidence] = []

    # --------------------------------------------------------------
    # 2. 症状反查
    #
    # 旧代码这里会：
    #
    # random.choice(res)
    #
    # 随机挑一个疾病塞进 entities。
    #
    # 这个行为非常不适合医疗系统：
    # 同一个症状可能对应很多疾病，不能随机选一个当成用户疾病。
    # --------------------------------------------------------------

    if (
        "疾病症状" in entities
        and "疾病" not in entities
    ):

        symptom = entities["疾病症状"]

        diseases = kg.get_diseases_by_symptom(
            symptom
        )

        if diseases:

            evidence_list.append(
                Evidence(
                    source_type="kg",
                    title="Neo4j 医疗知识图谱",
                    entity=symptom,
                    relation="疾病症状反向关联",
                    content="、".join(diseases),
                    note=(
                        "这些疾病仅与该症状存在图谱关联，"
                        "不能据此进行疾病诊断。"
                    ),
                )
            )

    # --------------------------------------------------------------
    # 3. KG Intent Retrieval
    # --------------------------------------------------------------

    intent_names, kg_evidence = (
        execute_intents_with_evidence(
            response=response,
            entities=entities,
            kg=kg,
        )
    )

    evidence_list.extend(
        kg_evidence
    )

    # --------------------------------------------------------------
    # 4. Evidence -> Grounded Prompt
    # --------------------------------------------------------------

    prompt = build_grounded_prompt(
        query=query,
        evidence_list=evidence_list,
    )

    logger.debug(
        "entities: %s",
        entities,
    )

    logger.debug(
        "intents: %s",
        intent_names,
    )

    logger.debug(
        "evidence: %s",
        [
            e.to_dict()
            for e in evidence_list
        ],
    )

    return (
        prompt,
        "、".join(intent_names),
        entities,
        evidence_list,
    )

def ans_stream(prompt):
    """[已弃用] 旧版 ChatGLM 流式回答接口。

    项目当前使用 ollama 通过 ``ollama.chat(..., stream=True)`` 在 ``main()`` 内直接
    流式输出，已不再依赖 ChatGLM；此函数仅作为历史占位保留为空实现，避免外部潜在引用
    报错。如需恢复 ChatGLM 流式回答，请实现一个接受 (model, tokenizer, prompt) 的版本。
    """
    raise NotImplementedError(
        "ans_stream 已弃用，请使用 main() 中的 ollama.chat 流式调用"
    )



def main(is_admin: bool, usname: str) -> None:
    """Streamlit 主界面入口；由 ``login.py`` 在用户登录成功后调用。"""
    medical_agent = load_medical_agent()

    cache_model = settings.NER_CHECKPOINT
    st.title("医疗智能问答机器人")

    # ==============================================================
    # 1. Sidebar
    # ==============================================================
    with st.sidebar:
        col1, _ = st.columns([0.6, 0.6])
        with col1:
            st.image(os.path.join("img", "logo.jpg"), use_column_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        # 初始化对话窗口
        if "chat_windows" not in st.session_state:
            st.session_state.chat_windows = [[]]

        if "messages" not in st.session_state:
            st.session_state.messages = [[]]

        if st.button("新建对话窗口"):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [
            f"对话窗口 {i + 1}"
            for i in range(len(st.session_state.chat_windows))
        ]
        selected_window = st.selectbox(
            "请选择对话窗口:",
            window_options,
        )
        active_window_index = int(selected_window.split()[1]) - 1

        selected_option = st.selectbox(
            label="请选择大语言模型:",
            options=["Qwen 1.5", "Llama2-Chinese"],
        )

        choice = (
            settings.OLLAMA_QWEN_MODEL
            if selected_option == "Qwen 1.5"
            else settings.OLLAMA_LLAMA_MODEL
        )

        show_ent = False
        show_int = False
        show_prompt = False
        show_evidence = False
        show_route = False

        if is_admin:
            show_ent = st.checkbox("显示实体识别结果")
            show_int = st.checkbox("显示意图识别结果")
            show_prompt = st.checkbox("显示查询的知识库信息")
            show_evidence = st.checkbox("显示结构化检索证据")
            show_route = st.checkbox("显示检索路由")

            if st.button("修改知识图谱"):
                st.markdown(
                    "[点击这里修改知识图谱](http://127.0.0.1:7474/)",
                    unsafe_allow_html=True,
                )

        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    # ==============================================================
    # 2. 加载 NER 模型 + Neo4j
    # ==============================================================
    (
        glm_tokenizer,
        glm_model,
        bert_tokenizer,
        bert_model,
        idx2tag,
        rule,
        tfidf_r,
        device,
    ) = load_model(cache_model)

    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    client = KGClient(graph)

    # Vector RAG Retriever
    vector_retriever = load_vector_retriever()

    current_messages = st.session_state.messages[active_window_index]

    # ==============================================================
    # 3. 回放历史消息
    # ==============================================================
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] != "assistant":
                continue

            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(message.get("ent", ""))

            if show_int:
                with st.expander("意图识别结果"):
                    st.write(message.get("yitu", ""))

            if show_prompt:
                with st.expander("点击显示知识库信息"):
                    prompt_text = message.get("prompt", "")
                    if prompt_text:
                        st.write(prompt_text)
                    else:
                        st.write("本轮没有有效知识库信息")

            if show_evidence:
                with st.expander("结构化检索证据"):
                    evidence_data = message.get("evidence", [])
                    if evidence_data:
                        st.json(evidence_data)
                    else:
                        st.write("本轮没有有效检索证据")


            if show_route:
                with st.expander("检索路由"):
                    st.write(
                        "retrieval_mode:",
                        message.get("retrieval_mode", ""),
                    )
                    st.write(
                        "route_reason:",
                        message.get("route_reason", ""),
                    )
                    st.write(
                        "answer_mode:",
                        message.get("answer_mode", ""),
                    )

    # ==============================================================
    # 4. 接收当前问题
    # ==============================================================
    query = st.chat_input(
        "Ask me anything!",
        key=f"chat_input_{active_window_index}",
    )

    if not query:
        return

    # 保存并显示用户消息
    current_messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # ==============================================================
    # 5. 特殊多轮请求：“你的依据是什么？”
    # ==============================================================
    if is_source_query(query):
        previous_turn = get_last_assistant_turn(
            current_messages[:-1]
        )

        if previous_turn is None:
            last = "当前对话中还没有可以追溯的上一轮回答。"
            evidence_data = []
        else:
            evidence_data = previous_turn.get(
                "evidence",
                [],
            )
            last = format_evidence_answer(
                evidence_data
            )

        with st.chat_message("assistant"):
            st.markdown(last)

            if show_evidence:
                with st.expander("结构化检索证据"):
                    if evidence_data:
                        st.json(evidence_data)
                    else:
                        st.write("上一轮没有保存有效检索证据")

        current_messages.append(
            {
                "role": "assistant",
                "content": last,
                "yitu": "证据追溯",
                "prompt": "",
                "ent": "",
                "evidence": evidence_data,
                "original_query": query,
                "resolved_query": query,
            }
        )

        st.session_state.messages[
            active_window_index
        ] = current_messages

        return

    # ==============================================================
    # 6. 普通问题 / 多轮追问
    # ==============================================================

    # --------------------------------------------------------------
    # 6.1 上下文改写
    # --------------------------------------------------------------
    with st.status(
        "正在处理问题...",
        expanded=False,
    ) as status:
        status.write("正在解析对话上下文...")

        resolved_query = resolve_followup_query(
            query=query,
            messages=current_messages[:-1],
            model=choice,
        )

        logger.info(
            "原始问题: %s | 上下文改写: %s",
            query,
            resolved_query,
        )

        # ----------------------------------------------------------
        # 6.2 Intent
        # ----------------------------------------------------------
        status.write("正在进行意图识别...")

        response = Intent_Recognition(
            resolved_query,
            choice,
        )

        # ----------------------------------------------------------
        # 6.3 Retrieval
        # ----------------------------------------------------------
        status.write("正在检索知识证据...")

        (
            prompt,
            yitu,
            entities,
            evidence_list,
        ) = generate_prompt(
            response,
            resolved_query,
            client,
            bert_model,
            bert_tokenizer,
            rule,
            tfidf_r,
            device,
            idx2tag,
        )

        # 当前 generate_prompt() 返回的是 KG Evidence。
        kg_evidence = list(evidence_list)

        (
            retrieval_mode,
            route_reason,
        ) = decide_retrieval_mode_v2(
            resolved_query,
            kg_evidence,
        )

        logger.info(
            "Retrieval Route: %s | reason=%s",
            retrieval_mode,
            route_reason,
        )

        status.write(
            f"检索路由: {retrieval_mode}"
        )

        vector_evidence: List[Evidence] = []

        if retrieval_mode in {
            "VECTOR",
            "HYBRID",
        }:
            if vector_retriever is not None and vector_retriever.ready():
                status.write(
                    "正在执行医学文档向量检索..."
                )

                try:
                    vector_evidence = (
                        vector_retriever.search(
                            resolved_query
                        )
                    )

                except Exception as exc:
                    logger.exception(
                        "Vector Retrieval 失败: %s",
                        exc,
                    )
                    vector_evidence = []

            else:
                logger.warning(
                    "Vector collection 尚未构建，"
                    "请先运行 build_vector_index.py"
                )

        # VECTOR 模式必须丢弃可能由错误 Intent 产生的 KG 证据。
        if retrieval_mode == "VECTOR":
            evidence_list = list(
                vector_evidence
            )

        elif retrieval_mode == "HYBRID":
            evidence_list = (
                list(kg_evidence)
                + list(vector_evidence)
            )

        else:
            evidence_list = list(
                kg_evidence
            )

        # KG + Vector 合并后重新构建 Grounded Prompt。
        # 纯 KG 情况后面仍由 kg_answer_formatter 确定性回答；
        # 只要存在 vector evidence，就会进入 LLM_GROUNDED。
        prompt = build_grounded_prompt(
            query=resolved_query,
            evidence_list=evidence_list,
        )

        status.update(
            label=(
                f"检索完成（{retrieval_mode}），"
                "正在生成回答..."
            ),
            state="complete",
            expanded=False,
        )

    # ==============================================================
    # 7. Answer Generation
    #
    # KG:
    #   确定性 Formatter，不调用 Qwen。
    #
    # VECTOR / HYBRID:
    #   必须真正命中 Vector Evidence 才允许 LLM 回答。
    # ==============================================================
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        if retrieval_mode == "KG":

            if not kg_evidence:
                answer_mode = "NO_EVIDENCE"
                last = (
                    "根据当前知识图谱证据无法回答该问题。"
                )
            else:
                answer_mode = "KG_DETERMINISTIC"
                last = format_kg_answer(
                    kg_evidence
                )

            response_placeholder.markdown(
                last
            )

        elif retrieval_mode in {
            "VECTOR",
            "HYBRID",
        }:

            if not vector_evidence:
                answer_mode = "NO_VECTOR_EVIDENCE"
                last = (
                    "当前医学文档知识库没有检索到足够相关的证据，"
                    "因此暂时无法基于可信文档完整回答该问题。"
                )

                response_placeholder.markdown(
                    last
                )

            else:
                answer_mode = "LLM_GROUNDED"
                last = ""

                try:
                    for chunk in ollama.chat(
                        model=choice,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        stream=True,
                        options={
                            "temperature": 0.1,
                            "seed": 42,
                        },
                    ):
                        content = chunk[
                            "message"
                        ][
                            "content"
                        ]

                        last += content

                        response_placeholder.markdown(
                            last
                        )

                except Exception as exc:
                    logger.exception(
                        "Ollama 回答生成失败: %s",
                        exc,
                    )

                    answer_mode = "LLM_ERROR"

                    last = (
                        "回答生成失败，请检查 Ollama 服务和模型状态。"
                    )

                    response_placeholder.error(
                        last
                    )

        else:
            answer_mode = "NO_EVIDENCE"
            last = (
                "根据当前知识库证据无法回答该问题。"
            )

            response_placeholder.markdown(
                last
            )

        # ----------------------------------------------------------
        # 7.1 结构化 Evidence
        # ----------------------------------------------------------
        evidence_data = [
            evidence.to_dict()
            for evidence in evidence_list
        ]

        # 兼容旧版“知识库信息”面板
        zhishiku_content = "\n\n".join(
            evidence.to_prompt(index)
            for index, evidence in enumerate(
                evidence_list,
                start=1,
            )
        )

        # ----------------------------------------------------------
        # 7.2 Debug 信息
        # ----------------------------------------------------------
        if show_ent:
            with st.expander("实体识别结果"):
                st.write(str(entities))

        if show_int:
            with st.expander("意图识别结果"):
                st.write(yitu)

        if show_prompt:
            with st.expander("点击显示知识库信息"):
                if zhishiku_content:
                    st.write(zhishiku_content)
                else:
                    st.write("没有命中有效知识证据")

        if show_evidence:
            with st.expander("结构化检索证据"):
                if evidence_data:
                    st.json(evidence_data)
                else:
                    st.write("本轮没有有效检索证据")


        if show_route:
            with st.expander("检索路由"):
                st.write(
                    "retrieval_mode:",
                    retrieval_mode,
                )
                st.write(
                    "route_reason:",
                    route_reason,
                )
                st.write(
                    "answer_mode:",
                    answer_mode,
                )
                st.write(
                    "KG Evidence 数:",
                    len(kg_evidence),
                )
                st.write(
                    "Vector Evidence 数:",
                    len(vector_evidence),
                )

    # ==============================================================
    # 8. 保存完整 Turn
    # ==============================================================
    current_messages.append(
        {
            "role": "assistant",
            "content": last,

            # 用户原始问题
            "original_query": query,

            # 上下文改写后的独立问题
            "resolved_query": resolved_query,

            # Pipeline 信息
            "yitu": yitu,
            "ent": str(entities),

            # 兼容旧 UI
            "prompt": zhishiku_content,

            # 新版核心字段
            "evidence": evidence_data,

            # 回答模式：
            # NO_EVIDENCE / KG_DETERMINISTIC / LLM_GROUNDED
            "answer_mode": answer_mode,

            # 检索路由：
            # KG / VECTOR / HYBRID
            "retrieval_mode": retrieval_mode,
            "route_reason": route_reason,
        }
    )

    st.session_state.messages[
        active_window_index
    ] = current_messages

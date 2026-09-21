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




# ======================================================================
# Modern Streamlit UI
# ======================================================================

APP_CSS = """
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --cyan: #06b6d4;
    --bg: #f5fafc;
    --card: #ffffff;
    --text: #172033;
    --muted: #7d8999;
    --line: #e4ebf3;
}

html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 74% 9%, rgba(14,165,233,.055), transparent 25%),
        linear-gradient(180deg, #f8fcfe 0%, #f4f9fc 100%);
    color: var(--text);
}

.main .block-container {
    max-width: 1120px;
    padding-top: 1.2rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,.97);
    border-right: 1px solid #e5edf5;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.7rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: rgba(248,252,254,.80);
    backdrop-filter: blur(9px);
}

/* 品牌 */
.brand-wrap {
    display:flex;
    align-items:center;
    gap:12px;
    margin: 2px 0 22px;
}
.brand-icon {
    width:44px;
    height:44px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#fff;
    font-size:23px;
    background:linear-gradient(135deg,#2563eb 0%,#06b6d4 100%);
    box-shadow:0 8px 18px rgba(37,99,235,.20);
}
.brand-name {
    font-size:18px;
    font-weight:760;
    color:#182033;
}
.brand-sub {
    font-size:12px;
    color:#8b96a6;
    margin-top:3px;
}

.side-section {
    color:#9aa5b5;
    font-size:12px;
    margin: 12px 0 8px;
}

/* 顶部状态栏 */
.top-assistant {
    display:flex;
    align-items:center;
    gap:12px;
    background:rgba(255,255,255,.86);
    border:1px solid #e8eef5;
    border-radius:14px;
    padding:12px 16px;
    margin-bottom:20px;
}
.top-avatar {
    width:40px;
    height:40px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#e0f2fe;
    color:#2563eb;
    font-size:20px;
}
.top-name {
    font-weight:760;
    color:#1f2937;
    font-size:16px;
}
.online-dot {
    display:inline-block;
    width:8px;
    height:8px;
    border-radius:50%;
    background:#22c55e;
    margin-right:6px;
}
.top-status {
    color:#7b8798;
    font-size:12px;
    margin-top:2px;
}

/* 欢迎区 */
.welcome-shell {
    text-align:center;
    padding:42px 10px 14px;
}
.welcome-icon {
    width:78px;
    height:78px;
    margin:0 auto 18px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:23px;
    color:#2563eb;
    background:linear-gradient(145deg,#dff4ff,#dbeafe);
    box-shadow:0 14px 30px rgba(37,99,235,.11);
    font-size:38px;
}
.welcome-title {
    font-size:27px;
    color:#1687bd;
    font-weight:760;
    margin-bottom:10px;
}
.welcome-subtitle {
    max-width:650px;
    margin:0 auto;
    color:#7b8798;
    font-size:14px;
    line-height:1.9;
}

.cap-card {
    background:#fff;
    border:1px solid #e5edf5;
    border-radius:14px;
    padding:18px 18px 16px;
    min-height:138px;
    box-shadow:0 8px 20px rgba(28,60,105,.035);
}
.cap-icon {
    width:36px;
    height:36px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#e8f6ff;
    color:#2563eb;
    font-size:18px;
    margin-bottom:12px;
}
.cap-title {
    font-size:16px;
    font-weight:740;
    color:#263244;
    margin-bottom:6px;
}
.cap-text {
    color:#8290a2;
    font-size:13px;
    line-height:1.65;
}

/* chat */
[data-testid="stChatMessage"] {
    background:#fff;
    border:1px solid #e7edf4;
    border-radius:15px;
    padding:10px 14px;
    margin-bottom:11px;
    box-shadow:0 4px 14px rgba(36,60,90,.025);
}

[data-testid="stChatInput"] {
    border-radius:14px;
}

[data-testid="stChatInput"] > div {
    background:#fff;
    border:1px solid #dce6f0;
    border-radius:14px;
    box-shadow:0 8px 25px rgba(31,65,114,.075);
}

/* 信息标签 */
.route-row {
    margin:12px 0 4px;
}
.metric-chip {
    display:inline-block;
    padding:6px 10px;
    margin:0 6px 6px 0;
    background:#f8fbff;
    border:1px solid #dfeaf5;
    border-radius:999px;
    color:#506073;
    font-size:12px;
}
.metric-chip strong {
    color:#1f5fbf;
}

/* 普通按钮 */
.stButton > button {
    border-radius:10px;
    border:1px solid #dce6f0;
    min-height:40px;
}

.stButton > button:hover {
    border-color:#93c5fd;
    color:#1d4ed8;
}

.sidebar-user {
    background:#f4f8fc;
    border:1px solid #e5edf5;
    border-radius:12px;
    padding:10px 12px;
    color:#596678;
    font-size:13px;
    margin-top:12px;
}

.disclaimer {
    text-align:center;
    color:#a0aaba;
    font-size:11px;
    margin-top:8px;
}
</style>
"""


def _evidence_source_counts(evidence_data):
    kg_count = 0
    vector_count = 0
    for item in evidence_data or []:
        source_type = str(item.get("source_type", "")).lower()
        if "kg" in source_type or "neo4j" in source_type:
            kg_count += 1
        elif "vector" in source_type or "document" in source_type:
            vector_count += 1
    return kg_count, vector_count


def _render_route_summary(
    retrieval_mode: str,
    answer_mode: str,
    evidence_data,
) -> None:
    kg_count, vector_count = _evidence_source_counts(evidence_data)
    st.markdown(
        f"""
        <div class="route-row">
            <span class="metric-chip">🔀 检索模式 <strong>{retrieval_mode or "-"}</strong></span>
            <span class="metric-chip">🕸 KG Evidence <strong>{kg_count}</strong></span>
            <span class="metric-chip">📄 Vector Evidence <strong>{vector_count}</strong></span>
            <span class="metric-chip">🤖 Answer Mode <strong>{answer_mode or "-"}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_evidence_list(evidence_data) -> None:
    if not evidence_data:
        st.caption("当前回答没有可展示的检索证据。")
        return

    for index, item in enumerate(evidence_data, start=1):
        source = item.get("source_type") or "unknown"
        title = item.get("title") or "未命名来源"
        content = item.get("content") or ""
        relation = item.get("relation")
        entity = item.get("entity")
        page = item.get("page")
        score = item.get("score")
        note = item.get("note")

        st.markdown(f"**Evidence {index} · `{source}`**")
        st.caption(title)

        meta = []
        if entity:
            meta.append(f"实体：{entity}")
        if relation:
            meta.append(f"关系/字段：{relation}")
        if page not in (None, ""):
            meta.append(f"页码：{page}")
        if isinstance(score, (int, float)):
            meta.append(f"Score：{score:.4f}")
        elif score not in (None, ""):
            meta.append(f"Score：{score}")

        if meta:
            st.caption(" · ".join(meta))

        st.info(content if content else "（无正文内容）")
        if note:
            st.caption(f"备注：{note}")

        if index != len(evidence_data):
            st.divider()


def _render_welcome() -> str | None:
    st.markdown(
        """
        <div class="welcome-shell">
            <div class="welcome-icon">⚕</div>
            <div class="welcome-title">您好，我是 MedEvidence AI</div>
            <div class="welcome-subtitle">
                面向医疗知识场景的多源 RAG 助手。系统会根据问题自动选择
                Knowledge Graph、Vector Retrieval 或 Hybrid Retrieval，
                并尽可能展示回答依据。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon">🕸</div>
                <div class="cap-title">知识图谱问答</div>
                <div class="cap-text">面向疾病、症状、药物、检查等结构化医学关系进行精确查询。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon">📄</div>
                <div class="cap-title">医学文档检索</div>
                <div class="cap-text">基于 Qdrant 向量检索与 Reranker，从医学文档中召回相关证据。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon">🔀</div>
                <div class="cap-title">多源智能路由</div>
                <div class="cap-text">根据问题自动选择 KG、Vector 或 Hybrid 路径，并进行 Evidence Grounding。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 可以试试这些问题")
    q1, q2, q3 = st.columns(3)
    quick_query = None
    with q1:
        if st.button("高血压有哪些症状？", use_container_width=True):
            quick_query = "高血压有哪些症状？"
    with q2:
        if st.button("为什么高血压需要长期管理？", use_container_width=True):
            quick_query = "为什么高血压需要长期管理？"
    with q3:
        if st.button("高血压有哪些症状，日常如何管理？", use_container_width=True):
            quick_query = "高血压有哪些症状，日常如何管理？"

    return quick_query


def main(is_admin: bool, usname: str) -> None:
    """Streamlit 主界面入口；保持原有 RAG 主链，仅重构展示层。"""

    # 当前 MedicalAgent/Memory 为扩展模块，未进入 Web 主问答链。
    # 不在这里提前初始化，避免与主 VectorRetriever 重复占用同一本地 Qdrant storage。
    cache_model = settings.NER_CHECKPOINT

    st.markdown(APP_CSS, unsafe_allow_html=True)

    # ==============================================================
    # 1. Sidebar
    # ==============================================================
    if "chat_windows" not in st.session_state:
        st.session_state.chat_windows = [[]]
    if "messages" not in st.session_state:
        st.session_state.messages = [[]]

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-icon">⚕</div>
                <div>
                    <div class="brand-name">MedEvidence AI</div>
                    <div class="brand-sub">医疗知识智能助手</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section">工作区</div>', unsafe_allow_html=True)

        if st.button("✚  新建会话", use_container_width=True):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])
            st.rerun()

        window_options = [
            f"会话 {i + 1}"
            for i in range(len(st.session_state.chat_windows))
        ]
        selected_window = st.selectbox(
            "会话记录",
            window_options,
            label_visibility="collapsed",
        )
        active_window_index = int(selected_window.split()[1]) - 1

        st.markdown('<div class="side-section">模型</div>', unsafe_allow_html=True)
        selected_option = st.selectbox(
            "生成模型",
            ["Qwen 1.5", "Llama2-Chinese"],
            label_visibility="collapsed",
        )
        choice = (
            settings.OLLAMA_QWEN_MODEL
            if selected_option == "Qwen 1.5"
            else settings.OLLAMA_LLAMA_MODEL
        )

        st.markdown('<div class="side-section">知识能力</div>', unsafe_allow_html=True)
        st.caption("🕸  Neo4j Knowledge Graph")
        st.caption("📄  Qdrant Vector Retrieval")
        st.caption("🎯  Cross Encoder Reranker")
        st.caption("🧾  Evidence Grounding")

        show_ent = False
        show_int = False
        show_prompt = False
        show_evidence = False
        show_route = False

        if is_admin:
            with st.expander("🛠 管理员调试面板"):
                show_ent = st.checkbox("实体识别结果")
                show_int = st.checkbox("意图识别结果")
                show_prompt = st.checkbox("知识库上下文")
                show_evidence = st.checkbox("结构化 Evidence JSON")
                show_route = st.checkbox("检索路由详情")
                st.markdown(
                    "[打开 Neo4j Browser](http://127.0.0.1:7474/)"
                )

        st.markdown("---")
        st.markdown(
            f"""
            <div class="sidebar-user">
                👤 <b>{usname}</b><br>
                <span style="color:#8b96a6;font-size:12px;">
                {'管理员' if is_admin else '普通用户'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("↪ 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.session_state.usname = ""
            st.rerun()

    # ==============================================================
    # 2. 顶部助手状态
    # ==============================================================
    st.markdown(
        """
        <div class="top-assistant">
            <div class="top-avatar">⚕</div>
            <div>
                <div class="top-name">MedEvidence AI · 医疗知识智能助手</div>
                <div class="top-status">
                    <span class="online-dot"></span>
                    在线 · KG / Vector / Hybrid 多源检索
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==============================================================
    # 3. 加载模型 + 数据库
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

    vector_retriever = load_vector_retriever()

    current_messages = st.session_state.messages[active_window_index]

    # ==============================================================
    # 4. 欢迎页
    # ==============================================================
    quick_query = None
    if not current_messages:
        quick_query = _render_welcome()

    # ==============================================================
    # 5. 回放历史消息
    # ==============================================================
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] != "assistant":
                continue

            evidence_data = message.get("evidence", [])
            _render_route_summary(
                message.get("retrieval_mode", ""),
                message.get("answer_mode", ""),
                evidence_data,
            )

            if evidence_data:
                with st.expander(
                    f"📚 查看回答依据（{len(evidence_data)} 条 Evidence）"
                ):
                    _render_evidence_list(evidence_data)

            if message.get("route_reason"):
                with st.expander("🔀 查看检索路由"):
                    st.markdown(
                        f"**Retrieval Mode：** `{message.get('retrieval_mode', '')}`"
                    )
                    st.markdown(
                        f"**Route Reason：** {message.get('route_reason', '')}"
                    )
                    st.markdown(
                        f"**Answer Mode：** `{message.get('answer_mode', '')}`"
                    )

            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(message.get("ent", ""))

            if show_int:
                with st.expander("意图识别结果"):
                    st.write(message.get("yitu", ""))

            if show_prompt:
                with st.expander("知识库上下文"):
                    st.write(message.get("prompt", "") or "本轮没有有效知识库信息")

            if show_evidence:
                with st.expander("Evidence JSON"):
                    st.json(evidence_data or [])

            if show_route:
                with st.expander("管理员路由详情"):
                    st.write("retrieval_mode:", message.get("retrieval_mode", ""))
                    st.write("route_reason:", message.get("route_reason", ""))
                    st.write("answer_mode:", message.get("answer_mode", ""))

    # ==============================================================
    # 6. 接收当前问题
    # ==============================================================
    input_query = st.chat_input(
        "请描述您的医疗知识问题，例如：高血压为什么需要长期管理？",
        key=f"chat_input_{active_window_index}",
    )
    query = quick_query or input_query

    st.markdown(
        '<div class="disclaimer">AI 生成内容仅用于知识参考，不替代专业医疗诊断与治疗建议。</div>',
        unsafe_allow_html=True,
    )

    if not query:
        return

    current_messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # ==============================================================
    # 7. 特殊多轮请求：“你的依据是什么？”
    # ==============================================================
    if is_source_query(query):
        previous_turn = get_last_assistant_turn(
            current_messages[:-1]
        )

        if previous_turn is None:
            last = "当前对话中还没有可以追溯的上一轮回答。"
            evidence_data = []
        else:
            evidence_data = previous_turn.get("evidence", [])
            last = format_evidence_answer(evidence_data)

        with st.chat_message("assistant"):
            st.markdown(last)
            if evidence_data:
                with st.expander(
                    f"📚 上一轮回答依据（{len(evidence_data)} 条 Evidence）"
                ):
                    _render_evidence_list(evidence_data)

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
                "answer_mode": "EVIDENCE_TRACE",
                "retrieval_mode": "TRACE",
                "route_reason": "用户请求追溯上一轮回答依据",
            }
        )

        st.session_state.messages[active_window_index] = current_messages
        return

    # ==============================================================
    # 8. 普通问题 / 多轮追问
    # ==============================================================
    with st.status(
        "正在进行多源知识检索...",
        expanded=False,
    ) as status:
        status.write("1/4 正在解析对话上下文...")

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

        status.write("2/4 正在进行意图识别与实体抽取...")

        response = Intent_Recognition(
            resolved_query,
            choice,
        )

        status.write("3/4 正在检索知识图谱证据...")

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
            f"4/4 路由到 {retrieval_mode}，正在补充检索证据..."
        )

        vector_evidence: List[Evidence] = []

        if retrieval_mode in {"VECTOR", "HYBRID"}:
            if vector_retriever is not None and vector_retriever.ready():
                try:
                    vector_evidence = vector_retriever.search(
                        resolved_query
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

        if retrieval_mode == "VECTOR":
            evidence_list = list(vector_evidence)
        elif retrieval_mode == "HYBRID":
            evidence_list = list(kg_evidence) + list(vector_evidence)
        else:
            evidence_list = list(kg_evidence)

        prompt = build_grounded_prompt(
            query=resolved_query,
            evidence_list=evidence_list,
        )

        status.update(
            label=f"检索完成 · {retrieval_mode}",
            state="complete",
            expanded=False,
        )

    # ==============================================================
    # 9. Answer Generation
    # ==============================================================
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        if retrieval_mode == "KG":
            if not kg_evidence:
                answer_mode = "NO_EVIDENCE"
                last = "根据当前知识图谱证据无法回答该问题。"
            else:
                answer_mode = "KG_DETERMINISTIC"
                last = format_kg_answer(kg_evidence)
            response_placeholder.markdown(last)

        elif retrieval_mode in {"VECTOR", "HYBRID"}:
            if not vector_evidence:
                answer_mode = "NO_VECTOR_EVIDENCE"
                last = (
                    "当前医学文档知识库没有检索到足够相关的证据，"
                    "因此暂时无法基于可信文档完整回答该问题。"
                )
                response_placeholder.markdown(last)
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
                        content = chunk["message"]["content"]
                        last += content
                        response_placeholder.markdown(last)

                except Exception as exc:
                    logger.exception(
                        "Ollama 回答生成失败: %s",
                        exc,
                    )
                    answer_mode = "LLM_ERROR"
                    last = "回答生成失败，请检查 Ollama 服务和模型状态。"
                    response_placeholder.error(last)

        else:
            answer_mode = "NO_EVIDENCE"
            last = "根据当前知识库证据无法回答该问题。"
            response_placeholder.markdown(last)

        evidence_data = [
            evidence.to_dict()
            for evidence in evidence_list
        ]

        zhishiku_content = "\n\n".join(
            evidence.to_prompt(index)
            for index, evidence in enumerate(
                evidence_list,
                start=1,
            )
        )

        # 面向普通用户直接展示“可解释检索”能力
        _render_route_summary(
            retrieval_mode,
            answer_mode,
            evidence_data,
        )

        if evidence_data:
            with st.expander(
                f"📚 查看回答依据（{len(evidence_data)} 条 Evidence）"
            ):
                _render_evidence_list(evidence_data)

        with st.expander("🔀 查看本轮检索路由"):
            st.markdown(
                f"**Retrieval Mode：** `{retrieval_mode}`"
            )
            st.markdown(
                f"**Route Reason：** {route_reason}"
            )
            st.markdown(
                f"**Answer Mode：** `{answer_mode}`"
            )
            st.caption(
                f"KG Evidence：{len(kg_evidence)} · "
                f"Vector Evidence：{len(vector_evidence)}"
            )

        # 管理员调试信息
        if show_ent:
            with st.expander("实体识别结果"):
                st.write(str(entities))

        if show_int:
            with st.expander("意图识别结果"):
                st.write(yitu)

        if show_prompt:
            with st.expander("知识库上下文"):
                st.write(
                    zhishiku_content
                    if zhishiku_content
                    else "没有命中有效知识证据"
                )

        if show_evidence:
            with st.expander("Evidence JSON"):
                st.json(evidence_data or [])

        if show_route:
            with st.expander("管理员路由详情"):
                st.write("retrieval_mode:", retrieval_mode)
                st.write("route_reason:", route_reason)
                st.write("answer_mode:", answer_mode)
                st.write("KG Evidence 数:", len(kg_evidence))
                st.write("Vector Evidence 数:", len(vector_evidence))

    # ==============================================================
    # 10. 保存完整 Turn
    # ==============================================================
    current_messages.append(
        {
            "role": "assistant",
            "content": last,
            "original_query": query,
            "resolved_query": resolved_query,
            "yitu": yitu,
            "ent": str(entities),
            "prompt": zhishiku_content,
            "evidence": evidence_data,
            "answer_mode": answer_mode,
            "retrieval_mode": retrieval_mode,
            "route_reason": route_reason,
        }
    )

    st.session_state.messages[active_window_index] = current_messages

"""意图路由：把 LLM 的意图识别输出映射为 KG 查询动作。

设计要点
========

* **表驱动**：16 类意图压缩成 :data:`INTENT_SPECS` 表，避免在 ``webui.py`` 里写 16 个重复的 ``if`` 块；
* **修复子串误匹配 bug**：原代码用 ``if "治疗" in response`` 会被「治疗周期」「治疗方法」同时命中，
  导致一次查询触发多个无关分支。本模块按关键字长度降序匹配，并用 ``set`` 记录已命中意图，
  确保「治疗周期」先匹配后「治疗方法」就不会再被「治疗」误触发。
* **意图标签与 prompt 对齐**：``intent_name`` 与 prompt 中定义的 16 类官方意图名严格对应，
  解决原代码 ``yitu.append('查询药物生产商')``（实际官方名是「查询药品的生产商」）等命名不一致问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kg_client import KGClient, build_attribute_prompt, build_relation_prompt
from evidence import Evidence

# ---------------------------------------------------------------------- spec


@dataclass(frozen=True)
class IntentSpec:
    """单个意图的查询规格。

    :ivar keywords: 触发关键字元组；只要 ``response`` 子串命中其中之一即视为该意图触发
    :ivar query_kind: ``"attribute"`` 或 ``"relation"`` 或 ``"symptom_lookup"``
    :ivar arg: 查询参数（属性名 / 关系名 / 症状反查时为空）
    :ivar target_label: 关系查询的目标节点标签；属性/症状查询时填 None
    :ivar intent_name: 用于回显的官方意图名
    :ivar entity_key: 从 NER 结果 entities 字典中取实体名时使用的 key（默认「疾病」）
    """

    keywords: Tuple[str, ...]
    query_kind: str
    arg: str
    target_label: Optional[str]
    intent_name: str
    entity_key: str = "疾病"


# 顺序：keywords 较长的需要排在前面以避免被短关键字误触发；
# 例如「治疗周期」必须排在「治疗」之前。下方表已按长度降序排列。
INTENT_SPECS: Tuple[IntentSpec, ...] = (
    # ---- 属性查询（疾病 → 单值属性） ----
    IntentSpec(("简介",),       "attribute", "疾病简介",    None, "查询疾病简介"),
    IntentSpec(("病因",),       "attribute", "疾病病因",    None, "查询疾病病因"),
    IntentSpec(("预防",),       "attribute", "预防措施",    None, "查询疾病预防措施"),
    IntentSpec(("治疗周期",),   "attribute", "治疗周期",    None, "查询疾病治疗周期"),
    IntentSpec(("治愈概率",),   "attribute", "治愈概率",    None, "查询治愈概率"),
    IntentSpec(("易感人群",),   "attribute", "疾病易感人群", None, "查询疾病易感人群"),
    # ---- 关系查询（疾病 → 多值实体） ----
    IntentSpec(("药品",),       "relation", "疾病使用药品", "药品",     "查询疾病所需药品"),
    IntentSpec(("宜吃食物",),   "relation", "疾病宜吃食物", "食物",     "查询疾病宜吃食物"),
    IntentSpec(("忌吃食物",),   "relation", "疾病忌吃食物", "食物",     "查询疾病忌吃食物"),
    IntentSpec(("检查项目",),   "relation", "疾病所需检查", "检查项目", "查询疾病所需检查项目"),
    IntentSpec(("查询疾病所属科目", "所属科目"),
                                "relation", "疾病所属科目", "科目",     "查询疾病所属科目"),
    # 注：「症状」必须排在「易感人群」/「治疗」等无关键字之后；
    # 「治疗方法」放在「治疗周期」之后以保证「治疗周期」先被匹配。
    IntentSpec(("治疗方法", "治疗"),
                                "relation", "治疗的方法",   "治疗方法", "查询疾病的治疗方法"),
    IntentSpec(("症状",),       "relation", "疾病的症状",   "疾病症状", "查询疾病的症状"),
    IntentSpec(("并发",),       "relation", "疾病并发疾病", "疾病",     "查询疾病的并发疾病"),
    # ---- 自定义查询（不依赖「疾病」实体） ----
    IntentSpec(("生产商",),     "drug_producer", "", None, "查询药品的生产商", entity_key="药品"),
)


# ---------------------------------------------------------------------- routing


def route_intents(response: str) -> List[IntentSpec]:
    """根据 LLM 意图识别输出文本，返回触发的意图规格列表。

    保证：每条意图最多触发一次；关键字按长度降序检测，避免子串误匹配。
    """
    matched: List[IntentSpec] = []
    seen: set[str] = set()
    # 把所有关键字按长度降序排列，绑定回各自所属的 spec
    keyword_to_spec: List[Tuple[str, IntentSpec]] = []
    for spec in INTENT_SPECS:
        for kw in spec.keywords:
            keyword_to_spec.append((kw, spec))
    keyword_to_spec.sort(key=lambda x: len(x[0]), reverse=True)

    # 用一个 mutable 副本，命中后把该关键字从 response 中划掉，避免被更短的同 spec 关键字重复匹配
    remaining = response
    for kw, spec in keyword_to_spec:
        if spec.intent_name in seen:
            continue
        if kw in remaining:
            matched.append(spec)
            seen.add(spec.intent_name)
            # 把命中的关键字替换为占位符，防止后续短关键字（如「治疗」）再次命中已处理过的「治疗周期」位置
            remaining = remaining.replace(kw, "\x00" * len(kw))
    return matched


def execute_intents(
    response: str,
    entities: Dict[str, str],
    kg: KGClient,
) -> Tuple[str, List[str]]:
    """根据 LLM 输出与 NER 实体，执行所有命中意图，返回 (拼接后的提示文本, 已命中意图名列表)。

    若意图所需实体不在 ``entities`` 中（例如询问药品生产商但 NER 未识别出药品），
    则跳过该意图但仍把意图名加入返回列表，以便 UI 透出「意图识别成功但缺少实体」的信息。
    """
    prompt_parts: List[str] = []
    intent_names: List[str] = []
    for spec in route_intents(response):
        entity = entities.get(spec.entity_key)
        intent_names.append(spec.intent_name)
        if not entity:
            continue
        if spec.query_kind == "attribute":
            value = kg.get_disease_attribute(entity, spec.arg)
            prompt_parts.append(build_attribute_prompt(entity, spec.arg, value))
        elif spec.query_kind == "relation":
            assert spec.target_label is not None
            items = kg.get_related_entities(entity, spec.arg, spec.target_label)
            prompt_parts.append(build_relation_prompt(entity, spec.arg, items))
        elif spec.query_kind == "drug_producer":
            items = kg.get_drug_producers(entity)
            prompt_parts.append(build_relation_prompt(entity, "药品生产商", items))
        # 未知 query_kind 静默跳过，不抛异常
    return "".join(prompt_parts), intent_names
def execute_intents_with_evidence(
    response: str,
    entities: Dict[str, str],
    kg: KGClient,
) -> Tuple[List[str], List[Evidence]]:
    """
    新版意图执行函数。

    与旧 execute_intents 最大区别：

    旧版：
        KG结果 -> 拼接 <提示>

    新版：
        KG结果 -> Evidence

    只有真正查询到数据时才产生 Evidence。
    """

    intent_names: List[str] = []
    evidence_list: List[Evidence] = []

    specs = route_intents(response)

    for spec in specs:

        intent_names.append(spec.intent_name)

        entity = entities.get(spec.entity_key)

        # 意图识别成功，但没有实体
        if not entity:
            continue

        # ----------------------------------------------------------
        # 1. 查询疾病属性
        # ----------------------------------------------------------
        if spec.query_kind == "attribute":

            value = kg.get_disease_attribute(
                entity,
                spec.arg,
            )

            # 没数据就不创建证据
            if not value:
                continue

            evidence_list.append(
                Evidence(
                    source_type="kg",
                    title="Neo4j 医疗知识图谱",
                    entity=entity,
                    relation=spec.arg,
                    content=str(value),
                    note=f"由意图「{spec.intent_name}」检索得到",
                )
            )

        # ----------------------------------------------------------
        # 2. 疾病关系查询
        # ----------------------------------------------------------
        elif spec.query_kind == "relation":

            if spec.target_label is None:
                continue

            items = kg.get_related_entities(
                entity,
                spec.arg,
                spec.target_label,
            )

            if not items:
                continue

            evidence_list.append(
                Evidence(
                    source_type="kg",
                    title="Neo4j 医疗知识图谱",
                    entity=entity,
                    relation=spec.arg,
                    content="、".join(items),
                    note=f"由意图「{spec.intent_name}」检索得到",
                )
            )

        # ----------------------------------------------------------
        # 3. 药物生产商
        # ----------------------------------------------------------
        elif spec.query_kind == "drug_producer":

            items = kg.get_drug_producers(entity)

            if not items:
                continue

            evidence_list.append(
                Evidence(
                    source_type="kg",
                    title="Neo4j 医疗知识图谱",
                    entity=entity,
                    relation="药品生产商",
                    content="、".join(items),
                    note=f"由意图「{spec.intent_name}」检索得到",
                )
            )

    return intent_names, evidence_list
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import List

from evidence import Evidence


def is_pure_kg_evidence(evidence_list: List[Evidence]) -> bool:
    """本轮存在证据且全部来自 Neo4j KG 时，使用确定性回答。"""
    return bool(evidence_list) and all(
        evidence.source_type == "kg"
        for evidence in evidence_list
    )


def _split_items(content: str) -> List[str]:
    """将 KG 多值字符串拆成去重后的列表。"""
    if not content:
        return []

    parts = re.split(r"[、，,；;\n]+", str(content))

    items: List[str] = []
    for part in parts:
        item = part.strip().strip(" |｜、，,；;")

        if not item:
            continue

        if item not in items:
            items.append(item)

    return items


def _bullet_list(items: List[str]) -> str:
    return "\n".join(
        f"- {item}"
        for item in items
    )


def _format_single_evidence(evidence: Evidence) -> str:
    """
    将一条 KG Evidence 转成确定性答案。

    注意：这里完全不调用 LLM，因此不会补充证据外医学事实。
    """
    entity = evidence.entity or "该实体"
    relation = evidence.relation or "知识图谱关系"
    content = str(evidence.content or "").strip()

    if not content:
        return ""

    items = _split_items(content)

    if relation == "疾病的症状":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 相关症状包括：\n\n"
            f"{body}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”关系。"
        )

    if relation == "治疗的方法":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 治疗方式包括：\n\n"
            f"{body}\n\n"
            f"以上内容仅表示知识图谱中记录的治疗方式类别，"
            f"没有补充图谱之外的具体治疗细节。"
        )

    if relation == "疾病使用药品":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱中与 **{entity}** 关联的药品包括：\n\n"
            f"{body}\n\n"
            f"以上仅表示 Neo4j 医疗知识图谱中的“{relation}”关联，"
            f"不代表具体处方或个体化用药建议。"
        )

    if relation == "疾病所需检查":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 相关检查项目包括：\n\n"
            f"{body}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”关系。"
        )

    if relation == "疾病所属科目":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 相关科室包括：\n\n"
            f"{body}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”关系。"
        )

    if relation == "疾病并发疾病":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 相关并发疾病包括：\n\n"
            f"{body}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”关系。"
        )

    if relation == "疾病宜吃食物":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱中与 **{entity}** 关联的宜吃食物包括：\n\n"
            f"{body}\n\n"
            f"以上仅表示当前知识图谱中的关联结果。"
        )

    if relation == "疾病忌吃食物":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱中与 **{entity}** 关联的忌吃食物包括：\n\n"
            f"{body}\n\n"
            f"以上仅表示当前知识图谱中的关联结果。"
        )

    if relation == "药品生产商":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱记录的 **{entity}** 生产商信息包括：\n\n"
            f"{body}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”关系。"
        )

    if relation == "疾病症状反向关联":
        body = _bullet_list(items) if items else content
        return (
            f"当前知识图谱中，与症状 **{entity}** 存在关联的疾病包括：\n\n"
            f"{body}\n\n"
            f"这些仅是知识图谱中的关联结果，不能据此进行疾病诊断。"
        )

    attribute_titles = {
        "疾病简介": "疾病简介",
        "疾病病因": "病因信息",
        "预防措施": "预防措施",
        "治疗周期": "治疗周期",
        "治愈概率": "治愈概率",
        "疾病易感人群": "易感人群",
    }

    if relation in attribute_titles:
        title = attribute_titles[relation]
        return (
            f"当前知识图谱记录的 **{entity}** {title}如下：\n\n"
            f"{content}\n\n"
            f"以上内容来自 Neo4j 医疗知识图谱中的“{relation}”字段。"
        )

    return (
        f"当前知识图谱返回了以下与 **{entity}** 相关的信息：\n\n"
        f"{content}\n\n"
        f"来源：Neo4j 医疗知识图谱；字段/关系：{relation}。"
    )


def format_kg_answer(
    evidence_list: List[Evidence],
) -> str:
    """
    组合本轮所有 KG Evidence。

    不调用 LLM，确保最终答案与 KG Evidence 一一对应。
    """
    if not evidence_list:
        return "根据当前知识库证据无法回答该问题。"

    answer_parts: List[str] = []

    for evidence in evidence_list:
        part = _format_single_evidence(evidence)

        if part:
            answer_parts.append(part)

    if not answer_parts:
        return "根据当前知识库证据无法回答该问题。"

    return "\n\n---\n\n".join(answer_parts)

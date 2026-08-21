# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

import ollama

from evidence import Evidence


SOURCE_QUERY_PATTERNS = (
    "你的依据是什么",
    "依据是什么",
    "你的依据",
    "你的证据是什么",
    "证据是什么",
    "来源是什么",
    "信息来源",
    "参考来源",
    "参考了什么",
    "参考什么",
    "出处是什么",
    "有什么依据",
    "有什么证据",
    "刚才的依据",
    "刚才的来源",
)


def is_source_query(query: str) -> bool:
    """
    判断当前问题是不是在追问上一轮回答的证据来源。
    """

    text = query.strip()

    return any(
        pattern in text
        for pattern in SOURCE_QUERY_PATTERNS
    )


def get_last_assistant_turn(
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    找到最近一次 assistant 回答。
    """

    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message

    return None


def format_evidence_answer(
    evidence_data: List[Dict[str, Any]],
) -> str:
    """
    把上一轮保存的结构化 Evidence 转成人能看的来源说明。
    """

    if not evidence_data:
        return (
            "上一轮回答没有保存可追溯的检索证据，"
            "因此目前无法可靠说明该回答的依据。"
        )

    evidence_list = [
        Evidence.from_dict(item)
        for item in evidence_data
    ]

    lines = [
        "上一轮回答使用了以下检索证据：",
        "",
    ]

    for index, evidence in enumerate(evidence_list, start=1):

        lines.append(f"**Evidence {index}**")
        lines.append(f"- 来源：{evidence.title}")

        if evidence.entity:
            lines.append(f"- 实体：{evidence.entity}")

        if evidence.relation:
            lines.append(
                f"- 图谱字段/关系：{evidence.relation}"
            )

        if evidence.section:
            lines.append(
                f"- 章节：{evidence.section}"
            )

        if evidence.page:
            lines.append(
                f"- 页码：{evidence.page}"
            )

        lines.append(
            f"- 内容：{evidence.content}"
        )

        lines.append("")

    # 目前 KG 数据没有文献级 provenance
    if all(
        evidence.source_type == "kg"
        for evidence in evidence_list
    ):
        lines.extend([
            "---",
            "",
            "需要说明：当前 Neo4j 医疗知识图谱只保存了"
            "疾病属性和实体关系，并没有保存每条知识对应的"
            "原始指南、论文、章节和页码。",
            "",
            "因此当前只能追溯到“知识图谱中的哪个字段/关系”，"
            "还不能做到真正的文献级引用。后续接入医学指南的"
            " Vector RAG 后，会进一步保存文档标题、章节、页码"
            "和检索片段。",
        ])

    return "\n".join(lines)


def resolve_followup_query(
    query: str,
    messages: List[Dict[str, Any]],
    model: str,
) -> str:
    """
    将多轮追问改写成完整的独立问题。

    示例：

    第一轮：
        高血压有什么症状？

    第二轮：
        这个病怎么治疗？

    改写为：
        高血压怎么治疗？
    """

    previous_turn = get_last_assistant_turn(messages)

    if previous_turn is None:
        return query

    previous_query = previous_turn.get(
        "resolved_query",
        ""
    )

    if not previous_query:
        return query

    prompt = f"""
你是一个对话查询改写器。

你的任务不是回答问题，而是判断“当前问题”是否依赖上一轮上下文。

上一轮完整问题：
{previous_query}

当前问题：
{query}

要求：

1. 如果当前问题已经是完整独立问题，原样返回。
2. 如果包含“它”“这个病”“这种情况”“还有呢”“为什么”
   等依赖上一轮上下文的表达，把它改写成一个完整独立问题。
3. 不允许回答医学问题。
4. 不允许增加上一轮中不存在的事实。
5. 只输出最终改写后的问题，不输出解释。

示例：

上一轮：高血压有什么症状？
当前：这个病怎么治疗？
输出：高血压怎么治疗？

上一轮：高血压有什么症状？
当前：糖尿病有什么症状？
输出：糖尿病有什么症状？

现在输出：
""".strip()

    try:
        result = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0,
                "seed": 42,
            },
        )

        rewritten = result["response"].strip()

        # 防止小模型返回过长解释
        if (
            not rewritten
            or len(rewritten) > 150
        ):
            return query

        return rewritten

    except Exception:
        return query
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """
    RAG 检索证据。

    当前 source_type 主要是：
    - kg: Neo4j 知识图谱
    - vector: 后续 Qdrant 文档检索
    """

    source_type: str
    title: str
    content: str

    entity: str = ""
    relation: str = ""

    # 为后续 Vector RAG 预留
    source_id: str = ""
    section: str = ""
    page: str = ""
    score: Optional[float] = None

    # 补充说明
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(**data)

    def to_prompt(self, index: int) -> str:
        """
        将证据转换成送给 LLM 的结构化 Evidence Block。
        """

        lines = [
            f"[Evidence {index}]",
            f"来源类型: {self.source_type}",
            f"来源: {self.title}",
        ]

        if self.entity:
            lines.append(f"实体: {self.entity}")

        if self.relation:
            lines.append(f"字段/关系: {self.relation}")

        if self.section:
            lines.append(f"章节: {self.section}")

        if self.page:
            lines.append(f"页码: {self.page}")

        if self.score is not None:
            lines.append(f"检索分数: {self.score:.4f}")

        if self.note:
            lines.append(f"说明: {self.note}")

        lines.append(f"证据内容: {self.content}")

        return "\n".join(lines)


def build_grounded_prompt(
    query: str,
    evidence_list: List[Evidence],
) -> str:
    """
    构建强 Grounding Prompt。

    原则：
    1. 医学事实只能来自 Evidence。
    2. Evidence 没有的信息不能自由补充。
    3. Evidence 不足就明确说证据不足。
    """

    if not evidence_list:
        return f"""
你是一个基于外部知识证据回答问题的医疗知识助手。

当前检索系统没有找到可以支持用户问题的有效证据。

用户问题：
{query}

要求：
只回答下面这一句话，不要补充任何医学知识：

根据当前知识库证据无法回答该问题。
""".strip()

    evidence_text = "\n\n".join(
        evidence.to_prompt(i)
        for i, evidence in enumerate(evidence_list, start=1)
    )

    return f"""
你是一个基于检索证据回答问题的医疗知识助手。

你的任务不是利用自己的参数知识自由回答，而是根据下面提供的 Evidence
整理出准确、简洁、可追溯的回答。

【严格规则】

1. 医学事实必须能够从 Evidence 中得到支持。
2. 不允许添加 Evidence 中不存在的具体疾病、药物、检查、治疗方式或数据。
3. 如果 Evidence 与用户问题只有部分相关，只回答能够被 Evidence 支持的部分。
4. 如果 Evidence 明显不足以回答核心问题，必须明确说明：
   “当前检索证据不足以完整回答该问题。”
5. 不要因为 Evidence 中出现了某项知识，就强行把它写进答案。
6. 优先回答用户真正询问的内容，而不是把所有 Evidence 全部复述一遍。
7. 回答最后增加：
   “证据：Evidence X”
   如果使用多条证据，可以写“Evidence 1、Evidence 2”。

【Evidence】

{evidence_text}

【用户问题】

{query}

请根据 Evidence 回答。
""".strip()
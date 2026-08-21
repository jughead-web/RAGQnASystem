# query_router.py
from typing import Any, Dict, List, Optional

# 这些意图默认更适合直接走知识图谱
KG_DIRECT_INTENTS = {
    "查询疾病简介",
    "查询疾病病因",
    "查询疾病的症状",
    "查询疾病预防措施",
    "查询疾病并发疾病",
    "查询疾病所需药品",
    "查询疾病的治疗方法",
}

# 这些关键词一旦命中，优先走 Vector / Hybrid
VECTOR_PRIORITY_KEYWORDS = [
    "为什么",
    "原因",
    "依据",
    "证据",
    "指南",
    "推荐",
    "首选",
    "长期",
    "生活方式",
    "干预",
    "控制目标",
    "目标血压",
    "达标",
    "随访",
    "复查",
    "监测",
    "多久",
    "频率",
    "管理",
    "注意事项",
    "是否需要长期",
    "开始药物治疗后",
]

# 这些问题通常是“解释型/规范型”问题，优先走 Hybrid
HYBRID_HINT_KEYWORDS = [
    "怎么管理",
    "如何管理",
    "如何干预",
    "治疗目标",
    "药物治疗后",
]

# 追问证据时，直接返回上一轮证据
TRACE_KEYWORDS = [
    "你的依据是什么",
    "依据是什么",
    "证据是什么",
    "你为什么这么说",
    "出处是什么",
    "来源是什么",
]


def _contains_any(text: str, keywords: List[str]) -> List[str]:
    text = text or ""
    return [kw for kw in keywords if kw in text]


def decide_route(
    question: str,
    entities: Optional[Dict[str, Any]] = None,
    intents: Optional[List[str]] = None,
    kg_evidences: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    返回:
    {
        "route": "KG" | "VECTOR" | "HYBRID" | "TRACE",
        "reason": "..."
    }
    """
    question = (question or "").strip()
    intents = intents or []
    kg_evidences = kg_evidences or []

    # 1. 先看是否在追问“依据/证据”
    trace_hits = _contains_any(question, TRACE_KEYWORDS)
    if trace_hits:
        return {
            "route": "TRACE",
            "reason": f"trace_keywords={trace_hits}"
        }

    # 2. 命中 Vector 优先关键词 => 优先走 Vector / Hybrid
    vector_hits = _contains_any(question, VECTOR_PRIORITY_KEYWORDS)
    if vector_hits:
        if kg_evidences:
            return {
                "route": "HYBRID",
                "reason": f"vector_priority_keywords={vector_hits}"
            }
        return {
            "route": "VECTOR",
            "reason": f"vector_priority_keywords={vector_hits}"
        }

    # 3. 命中 KG 直接意图，且确实有 KG 证据 => 走 KG
    kg_hits = [x for x in intents if x in KG_DIRECT_INTENTS]
    if kg_hits and kg_evidences:
        return {
            "route": "KG",
            "reason": f"kg_direct_intents={kg_hits}"
        }

    # 4. 解释型/规范型问题 => Hybrid / Vector
    hybrid_hits = _contains_any(question, HYBRID_HINT_KEYWORDS)
    if hybrid_hits:
        if kg_evidences:
            return {
                "route": "HYBRID",
                "reason": f"hybrid_hint_keywords={hybrid_hits}"
            }
        return {
            "route": "VECTOR",
            "reason": f"hybrid_hint_keywords={hybrid_hits}"
        }

    # 5. fallback
    if kg_evidences:
        return {
            "route": "KG",
            "reason": "fallback_has_kg"
        }

    return {
        "route": "VECTOR",
        "reason": "fallback_no_kg"
    }
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from vector_rag import VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "单独测试 Dense Retrieval + Reranker，"
            "无需启动 Streamlit"
        )
    )

    parser.add_argument(
        "query",
        help="测试问题",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    retriever = VectorRetriever()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        candidate_top_k=(
            args.candidate_top_k
        ),
    )

    print()
    print("=" * 80)
    print(
        "Query:",
        args.query,
    )
    print("=" * 80)

    if not results:
        print("没有检索到 Evidence")
        return

    for index, evidence in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"[FINAL RANK {index}]"
        )
        print(
            "title:",
            evidence.title,
        )
        print(
            "page:",
            evidence.page,
        )
        print(
            "rerank_score:",
            evidence.score,
        )
        print(
            "note:",
            evidence.note,
        )
        print(
            "content:",
            evidence.content[:800],
        )
        print("-" * 80)


if __name__ == "__main__":
    main()

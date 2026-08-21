# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from vector_rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    VectorIndexBuilder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "重建医学文档 Qdrant 索引："
            "包含 PDF 清洗 + Chunk + Embedding"
        )
    )

    parser.add_argument(
        "--docs",
        default="medical_docs",
    )

    parser.add_argument(
        "--qdrant-path",
        default=QDRANT_PATH,
    )

    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
    )

    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("Vector RAG Index Rebuild V2")
    print("包含：PDF Cleaning + Dense Index")
    print(f"文档目录: {args.docs}")
    print(
        f"Qdrant目录: "
        f"{args.qdrant_path}"
    )
    print(
        f"Collection: "
        f"{args.collection}"
    )
    print(
        f"Embedding: "
        f"{args.model}"
    )
    print("=" * 72)

    builder = VectorIndexBuilder(
        qdrant_path=(
            args.qdrant_path
        ),
        collection_name=(
            args.collection
        ),
        model_name=args.model,
    )

    count = builder.rebuild(
        docs_dir=args.docs
    )

    print()
    print("=" * 72)
    print("构建完成")
    print(
        f"成功写入 chunk 数: "
        f"{count}"
    )
    print("=" * 72)
    print(
        "注意：Reranker 不需要建入 Qdrant；"
        "它会在 Streamlit 运行时加载，"
        "对 Dense Retrieval 候选进行重排序。"
    )


if __name__ == "__main__":
    main()

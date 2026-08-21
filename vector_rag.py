# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from evidence import Evidence
from pdf_text_cleaner import clean_pdf_pages


# ======================================================================
# 基础配置
# ======================================================================

EMBEDDING_MODEL = os.getenv(
    "VECTOR_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
)

# FastEmbed 官方支持的 BGE Cross-Encoder Reranker。
# 这是中文场景优先选项，同时为 MIT License。
RERANKER_MODEL = os.getenv(
    "VECTOR_RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "qdrant_storage",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "medical_guidelines",
)

# 第一阶段 Dense Retrieval 召回多少候选
VECTOR_CANDIDATE_TOP_K = int(
    os.getenv(
        "VECTOR_CANDIDATE_TOP_K",
        "20",
    )
)

# Reranker 最终保留多少 Evidence
VECTOR_TOP_K = int(
    os.getenv(
        "VECTOR_TOP_K",
        "4",
    )
)

# 第一阶段为了保证 Recall，不要把阈值设得太高
VECTOR_SCORE_THRESHOLD = float(
    os.getenv(
        "VECTOR_SCORE_THRESHOLD",
        "0.30",
    )
)

CHUNK_SIZE = int(
    os.getenv(
        "VECTOR_CHUNK_SIZE",
        "420",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "VECTOR_CHUNK_OVERLAP",
        "90",
    )
)


@dataclass
class RawDocument:
    title: str
    text: str
    source: str
    section: str = ""
    page: Optional[int] = None


@dataclass
class TextChunk:
    chunk_id: str
    title: str
    text: str
    source: str
    section: str = ""
    page: Optional[int] = None


@dataclass
class DenseCandidate:
    title: str
    text: str
    source: str
    section: str
    page: Optional[int]
    chunk_id: str
    dense_score: float
    dense_rank: int


# ======================================================================
# 文本处理
# ======================================================================

def _normalize_text(text: str) -> str:
    """
    TXT/Markdown 的轻量清洗。
    PDF 使用专门的 pdf_text_cleaner。
    """
    text = text.replace(
        "\u3000",
        " ",
    )
    text = text.replace(
        "\xa0",
        " ",
    )
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )
    return text.strip()


def _split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    中文轻量分块。

    - chunk_size 默认 420 字符；
    - overlap 默认 90；
    - 优先在完整句子/段落边界切断。
    """
    text = _normalize_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        hard_end = min(
            start + chunk_size,
            n,
        )

        if hard_end == n:
            end = n
        else:
            search_start = (
                start
                + max(
                    100,
                    chunk_size // 2,
                )
            )

            candidate = text[
                search_start:hard_end
            ]

            cut_positions = [
                candidate.rfind("\n"),
                candidate.rfind("。"),
                candidate.rfind("！"),
                candidate.rfind("？"),
                candidate.rfind("；"),
                candidate.rfind(". "),
                candidate.rfind("? "),
                candidate.rfind("! "),
            ]

            best = max(
                cut_positions
            )

            if best >= 0:
                end = (
                    search_start
                    + best
                    + 1
                )
            else:
                end = hard_end

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = max(
            0,
            end - overlap,
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ======================================================================
# 文件解析
# ======================================================================

def _load_pdf(
    path: Path,
) -> List[RawDocument]:
    """
    PDF：
        pypdf 原始抽取
            ↓
        整文档页眉/页脚检测
            ↓
        bullet / 乱码 / 视觉换行清洗
            ↓
        保留真实 PDF 页码
    """
    reader = PdfReader(
        str(path)
    )

    title = path.stem

    raw_pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    (
        cleaned_pages,
        stats,
    ) = clean_pdf_pages(
        raw_pages
    )

    print(
        f"[PDF CLEAN] {path.name} | "
        f"pages={stats.pages} | "
        f"header/footer_removed="
        f"{stats.repeated_header_footer_lines} | "
        f"noise_removed="
        f"{stats.dropped_noise_lines} | "
        f"bullets_repaired="
        f"{stats.repaired_bullet_lines}"
    )

    docs: List[RawDocument] = []

    for page_index, text in enumerate(
        cleaned_pages,
        start=1,
    ):
        if not text.strip():
            continue

        docs.append(
            RawDocument(
                title=title,
                text=text,
                source=str(path),
                page=page_index,
            )
        )

    return docs


def _load_txt(
    path: Path,
) -> List[RawDocument]:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        RawDocument(
            title=path.stem,
            text=_normalize_text(text),
            source=str(path),
        )
    ]


def _load_markdown(
    path: Path,
) -> List[RawDocument]:
    """
    Markdown 按标题拆 section。
    """
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    title = path.stem
    current_section = ""
    buffer: List[str] = []
    docs: List[RawDocument] = []

    def flush() -> None:
        nonlocal buffer

        body = _normalize_text(
            "\n".join(buffer)
        )

        if body:
            docs.append(
                RawDocument(
                    title=title,
                    text=body,
                    source=str(path),
                    section=current_section,
                )
            )

        buffer = []

    for line in text.splitlines():
        match = re.match(
            r"^\s*#{1,6}\s+(.+?)\s*$",
            line,
        )

        if match:
            flush()

            current_section = (
                match.group(1)
                .strip()
            )

            if line.lstrip().startswith(
                "# "
            ):
                title = current_section

        else:
            buffer.append(line)

    flush()
    return docs


def load_documents(
    docs_dir: str | Path,
) -> List[RawDocument]:
    root = Path(docs_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"医学文档目录不存在: {root}"
        )

    docs: List[RawDocument] = []

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                docs.extend(
                    _load_pdf(path)
                )

            elif suffix in {
                ".md",
                ".markdown",
            }:
                docs.extend(
                    _load_markdown(path)
                )

            elif suffix == ".txt":
                docs.extend(
                    _load_txt(path)
                )

        except Exception as exc:
            print(
                f"[WARN] 读取失败: "
                f"{path} -> {exc}"
            )

    return [
        doc
        for doc in docs
        if doc.text.strip()
    ]


def build_chunks(
    documents: Iterable[RawDocument],
) -> List[TextChunk]:
    chunks: List[TextChunk] = []

    for doc in documents:
        parts = _split_text(
            doc.text
        )

        for index, part in enumerate(
            parts
        ):
            raw_id = (
                f"{doc.source}|"
                f"{doc.page}|"
                f"{doc.section}|"
                f"{index}|"
                f"{part}"
            )

            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    raw_id,
                )
            )

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    title=doc.title,
                    text=part,
                    source=doc.source,
                    section=doc.section,
                    page=doc.page,
                )
            )

    return chunks


# ======================================================================
# Index Builder
# ======================================================================

class VectorIndexBuilder:
    """
    构建 Qdrant Dense Vector Index。

    注意：
    Reranker 不参与建库。
    Reranker 只在运行时对 Dense Retrieval 候选重排。
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        self.qdrant_path = (
            qdrant_path
        )
        self.collection_name = (
            collection_name
        )
        self.model_name = (
            model_name
        )

        self.embedding_model = (
            TextEmbedding(
                model_name=model_name
            )
        )

        self.client = QdrantClient(
            path=qdrant_path,
        )

    def rebuild(
        self,
        docs_dir: str | Path,
    ) -> int:
        documents = load_documents(
            docs_dir
        )

        chunks = build_chunks(
            documents
        )

        if not chunks:
            raise RuntimeError(
                "没有读取到可索引的医学文档。"
            )

        print(
            f"[INDEX] documents/pages="
            f"{len(documents)}"
        )
        print(
            f"[INDEX] chunks="
            f"{len(chunks)}"
        )

        vectors = list(
            self.embedding_model.passage_embed(
                [
                    chunk.text
                    for chunk in chunks
                ]
            )
        )

        vector_size = int(
            len(vectors[0])
        )

        if self.client.collection_exists(
            self.collection_name
        ):
            self.client.delete_collection(
                self.collection_name
            )

        self.client.create_collection(
            collection_name=(
                self.collection_name
            ),
            vectors_config=(
                models.VectorParams(
                    size=vector_size,
                    distance=(
                        models.Distance.COSINE
                    ),
                )
            ),
        )

        batch_size = 64

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            chunk_batch = chunks[
                start:start + batch_size
            ]

            vector_batch = vectors[
                start:start + batch_size
            ]

            points = []

            for chunk, vector in zip(
                chunk_batch,
                vector_batch,
            ):
                payload = {
                    "text": chunk.text,
                    "title": chunk.title,
                    "source": chunk.source,
                    "section": chunk.section,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "embedding_model": (
                        self.model_name
                    ),
                    "cleaning_version": "v2",
                }

                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector=(
                            vector.tolist()
                        ),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )

        return len(chunks)


# ======================================================================
# Retriever + Reranker
# ======================================================================

class VectorRetriever:
    """
    两阶段检索：

    Query
      ↓
    Dense Retrieval Top 20
      ↓
    BGE Cross-Encoder Reranker
      ↓
    Final Top 4 Evidence
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        score_threshold: float = VECTOR_SCORE_THRESHOLD,
    ) -> None:
        self.collection_name = (
            collection_name
        )
        self.model_name = (
            model_name
        )
        self.reranker_model_name = (
            reranker_model
        )
        self.score_threshold = (
            score_threshold
        )

        self.embedding_model = (
            TextEmbedding(
                model_name=model_name
            )
        )

        print(
            "[RERANKER] loading: "
            f"{reranker_model}"
        )

        self.reranker = TextCrossEncoder(
            model_name=reranker_model
        )

        self.client = QdrantClient(
            path=qdrant_path,
            force_disable_check_same_thread=True,
        )

    def ready(self) -> bool:
        return self.client.collection_exists(
            self.collection_name
        )

    def _dense_search(
        self,
        query: str,
        candidate_top_k: int,
    ) -> List[DenseCandidate]:
        query_vector = list(
            self.embedding_model.query_embed(
                query
            )
        )[0]

        response = self.client.query_points(
            collection_name=(
                self.collection_name
            ),
            query=query_vector.tolist(),
            limit=candidate_top_k,
            with_payload=True,
            score_threshold=(
                self.score_threshold
            ),
        )

        candidates: List[
            DenseCandidate
        ] = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = point.payload or {}

            text = str(
                payload.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            candidates.append(
                DenseCandidate(
                    title=str(
                        payload.get(
                            "title",
                            "医学文档",
                        )
                    ),
                    text=text,
                    source=str(
                        payload.get(
                            "source",
                            "",
                        )
                    ),
                    section=str(
                        payload.get(
                            "section",
                            "",
                        )
                    ),
                    page=(
                        payload.get(
                            "page"
                        )
                    ),
                    chunk_id=str(
                        payload.get(
                            "chunk_id",
                            point.id,
                        )
                    ),
                    dense_score=float(
                        point.score
                    ),
                    dense_rank=rank,
                )
            )

        return candidates

    def _rerank(
        self,
        query: str,
        candidates: List[DenseCandidate],
        final_top_k: int,
    ) -> List[tuple[DenseCandidate, float]]:
        if not candidates:
            return []

        documents = [
            candidate.text
            for candidate in candidates
        ]

        rerank_scores = list(
            self.reranker.rerank(
                query,
                documents,
            )
        )

        ranking = list(
            zip(
                candidates,
                [
                    float(score)
                    for score in rerank_scores
                ],
            )
        )

        ranking.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ranking[
            :final_top_k
        ]

    def search(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K,
        candidate_top_k: int = VECTOR_CANDIDATE_TOP_K,
    ) -> List[Evidence]:
        """
        返回已经 rerank 后的 Evidence。

        Evidence.score:
            reranker score

        Evidence.note:
            同时记录 dense rank / dense score / rerank rank。
        """
        if not self.ready():
            return []

        candidate_top_k = max(
            candidate_top_k,
            top_k,
        )

        candidates = (
            self._dense_search(
                query=query,
                candidate_top_k=(
                    candidate_top_k
                ),
            )
        )

        if not candidates:
            return []

        reranked = self._rerank(
            query=query,
            candidates=candidates,
            final_top_k=top_k,
        )

        evidence_list: List[
            Evidence
        ] = []

        for rerank_rank, (
            candidate,
            rerank_score,
        ) in enumerate(
            reranked,
            start=1,
        ):
            page_value = (
                candidate.page
            )

            evidence_list.append(
                Evidence(
                    source_type="vector",
                    title=candidate.title,
                    content=candidate.text,
                    source_id=(
                        candidate.source
                    ),
                    section=(
                        candidate.section
                    ),
                    page=(
                        str(page_value)
                        if page_value
                        is not None
                        else ""
                    ),
                    # 最终 score 使用 reranker
                    score=rerank_score,
                    relation="语义检索+重排序",
                    note=(
                        f"Rerank Rank "
                        f"{rerank_rank}; "
                        f"Dense Rank "
                        f"{candidate.dense_rank}; "
                        f"Dense Score "
                        f"{candidate.dense_score:.4f}; "
                        f"Reranker "
                        f"{self.reranker_model_name}"
                    ),
                )
            )

        return evidence_list

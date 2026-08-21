# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from evidence import Evidence
from pdf_text_cleaner import clean_pdf_pages

COLLECTION = os.getenv(
    "GLOBAL_MEDICAL_COLLECTION",
    "global_medical_kb_e5"
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "qdrant_storage"
)

EMBED_MODEL = os.getenv(
    "VECTOR_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
)

RERANK_MODEL = os.getenv(
    "VECTOR_RERANKER_MODEL",
    "BAAI/bge-reranker-base"
)
CHUNK_SIZE = 420
CHUNK_OVERLAP = 90
DENSE_TOP_K = 20
FINAL_TOP_K = 4
DENSE_THRESHOLD = 0.30


def load_registry(path: str = "global_sources.json") -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [x for x in data.get("sources", []) if x.get("enabled", True)]
    seen = set()
    for item in items:
        required = [
            "doc_id", "filename", "title", "publisher", "year",
            "language", "source_type", "diseases", "topics",
            "source_url", "authority_level",
        ]
        miss = [k for k in required if k not in item]
        if miss:
            raise ValueError(f"{item.get('filename')} 缺字段: {miss}")
        if item["doc_id"] in seen:
            raise ValueError(f"doc_id 重复: {item['doc_id']}")
        seen.add(item["doc_id"])
    return items


def _normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str) -> List[str]:
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks, start, n = [], 0, len(text)
    while start < n:
        hard_end = min(start + CHUNK_SIZE, n)
        if hard_end == n:
            end = n
        else:
            s = start + max(100, CHUNK_SIZE // 2)
            sub = text[s:hard_end]
            pos = max(
                sub.rfind("\n"), sub.rfind("。"), sub.rfind("！"),
                sub.rfind("？"), sub.rfind("；"), sub.rfind(". ")
            )
            end = s + pos + 1 if pos >= 0 else hard_end

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break

        nxt = max(0, end - CHUNK_OVERLAP)
        start = end if nxt <= start else nxt

    return chunks


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_pdf(path: Path) -> List[str]:
    reader = PdfReader(str(path))
    raw = [page.extract_text() or "" for page in reader.pages]
    cleaned, stats = clean_pdf_pages(raw)
    print(
        f"[PDF CLEAN] {path.name} | pages={stats.pages} | "
        f"header/footer_removed={stats.repeated_header_footer_lines} | "
        f"noise_removed={stats.dropped_noise_lines} | "
        f"bullets_repaired={stats.repaired_bullet_lines}"
    )
    return cleaned


class GlobalKBBuilder:
    def __init__(self, qdrant_path: str = QDRANT_PATH, collection: str = COLLECTION):
        self.collection = collection
        self.client = QdrantClient(
            path=qdrant_path,
            force_disable_check_same_thread=True,
        )
        self.embed = TextEmbedding(model_name=EMBED_MODEL)

    def _ensure_collection(self, rebuild: bool) -> None:
        exists = self.client.collection_exists(self.collection)
        if rebuild and exists:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return

        dim = len(list(self.embed.query_embed("医学知识"))[0])
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    def _delete_doc(self, doc_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def build(
        self,
        docs_dir: str = "medical_docs",
        registry_path: str = "global_sources.json",
        rebuild: bool = False,
    ) -> Dict[str, int]:
        sources = load_registry(registry_path)
        docs_root = Path(docs_dir)
        self._ensure_collection(rebuild)

        total_pages = 0
        total_chunks = 0

        for meta in sources:
            path = docs_root / meta["filename"]
            if not path.exists():
                raise FileNotFoundError(f"注册文档不存在: {path}")

            pages = _load_pdf(path)
            records = []

            for page_no, page_text in enumerate(pages, start=1):
                if not page_text.strip():
                    continue
                total_pages += 1

                for chunk_idx, text in enumerate(_split(page_text)):
                    content_hash = _hash(text)
                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{meta['doc_id']}|{page_no}|{chunk_idx}|{content_hash}",
                        )
                    )
                    payload = {
                        **meta,
                        "page": page_no,
                        "chunk_index": chunk_idx,
                        "text": text,
                        "content_hash": content_hash,
                        "kb_layer": "global_medical",
                        "local_path": str(path),
                    }
                    records.append((point_id, text, payload))

            if not rebuild:
                self._delete_doc(meta["doc_id"])

            vectors = list(self.embed.passage_embed([x[1] for x in records]))
            points = [
                models.PointStruct(id=pid, vector=vec.tolist(), payload=payload)
                for (pid, _, payload), vec in zip(records, vectors)
            ]

            for i in range(0, len(points), 64):
                self.client.upsert(
                    collection_name=self.collection,
                    points=points[i:i+64],
                    wait=True,
                )

            total_chunks += len(records)
            print(
                f"[OK] {meta['doc_id']} | pages={len([p for p in pages if p.strip()])} "
                f"| chunks={len(records)}"
            )

        return {
            "sources": len(sources),
            "pages": total_pages,
            "chunks": total_chunks,
        }


def _make_filter(
    disease: Optional[str] = None,
    source_type: Optional[str] = None,
    publisher: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Optional[models.Filter]:
    must = []

    if disease:
        must.append(
            models.FieldCondition(
                key="diseases",
                match=models.MatchAny(any=[disease]),
            )
        )
    if source_type:
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value=source_type),
            )
        )
    if publisher:
        must.append(
            models.FieldCondition(
                key="publisher",
                match=models.MatchValue(value=publisher),
            )
        )
    if doc_id:
        must.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        )

    return models.Filter(must=must) if must else None


class GlobalMedicalRetriever:
    def __init__(self, qdrant_path: str = QDRANT_PATH, collection: str = COLLECTION):
        self.collection = collection
        self.client = QdrantClient(
            path=qdrant_path,
            force_disable_check_same_thread=True,
        )
        self.embed = TextEmbedding(model_name=EMBED_MODEL)
        self.reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    def ready(self) -> bool:
        return self.client.collection_exists(self.collection)

    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        candidate_top_k: int = DENSE_TOP_K,
        disease: Optional[str] = None,
        source_type: Optional[str] = None,
        publisher: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Evidence]:
        if not self.ready():
            return []

        qvec = list(
            self.embed.query_embed(
                "query: " + query
                )
            )[0]

        response = self.client.query_points(
            collection_name=self.collection,
            query=qvec.tolist(),
            query_filter=_make_filter(
                disease=disease,
                source_type=source_type,
                publisher=publisher,
                doc_id=doc_id,
            ),
            limit=max(top_k, candidate_top_k),
            with_payload=True,
            score_threshold=DENSE_THRESHOLD,
        )

        candidates = []
        for dense_rank, point in enumerate(response.points, start=1):
            p = point.payload or {}
            text = str(p.get("text", "")).strip()
            if text:
                candidates.append((p, dense_rank, float(point.score)))

        if not candidates:
            return []

        docs = [x[0]["text"] for x in candidates]
        rerank_scores = [float(x) for x in self.reranker.rerank(query, docs)]

        ranked = list(zip(candidates, rerank_scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        out: List[Evidence] = []
        for rerank_rank, ((p, dense_rank, dense_score), rerank_score) in enumerate(
            ranked[:top_k], start=1
        ):
            note = (
                f"publisher={p.get('publisher')}; year={p.get('year')}; "
                f"source_type={p.get('source_type')}; authority={p.get('authority_level')}; "
                f"dense_rank={dense_rank}; dense_score={dense_score:.4f}; "
                f"rerank_rank={rerank_rank}; source_url={p.get('source_url')}"
            )
            out.append(
                Evidence(
                    source_type="vector",
                    title=str(p.get("title", "医学文档")),
                    content=str(p.get("text", "")),
                    source_id=str(p.get("doc_id", "")),
                    section="",
                    page=p.get("page"),
                    score=rerank_score,
                    relation="全局医学知识检索",
                    note=note,
                )
            )

        return out
    def close(self):
        try:
            self.client.close()
        except:
            pass
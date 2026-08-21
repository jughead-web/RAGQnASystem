# -*- coding: utf-8 -*-
import argparse
from global_medical_kb import GlobalMedicalRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    r = GlobalMedicalRetriever()
    if not r.ready():
        print("请先运行: python build_global_kb.py --rebuild")
        return

    results = r.search(
        args.query,
        top_k=args.top_k,
        disease=args.disease,
        source_type=args.source_type,
    )

    for i, e in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print(f"[EVIDENCE {i}]")
        print("title :", e.title)
        print("page  :", e.page)
        print("score :", e.score)
        print("doc_id:", e.source_id)
        print("note  :", e.note)
        print("content:")
        print(e.content[:1200])

if __name__ == "__main__":
    main()

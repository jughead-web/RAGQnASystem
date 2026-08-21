# -*- coding: utf-8 -*-
import argparse
from global_medical_kb import GlobalKBBuilder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="medical_docs")
    parser.add_argument("--registry", default="global_sources.json")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    result = GlobalKBBuilder().build(
        docs_dir=args.docs,
        registry_path=args.registry,
        rebuild=args.rebuild,
    )

    print("=" * 72)
    print("Global Medical KB 构建完成")
    print("权威来源数:", result["sources"])
    print("有效页数:", result["pages"])
    print("Chunk 数:", result["chunks"])
    print("=" * 72)

if __name__ == "__main__":
    main()

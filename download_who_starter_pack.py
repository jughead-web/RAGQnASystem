# -*- coding: utf-8 -*-
"""
WHO 医学资料启动包下载器。

说明：
1. 只从 WHO / WHO IRIS 官方地址下载。
2. 如果 WHO 对脚本下载返回 403，会自动提示你打开官方页面手动下载。
3. 下载成功后，把本脚本生成的 PDF 保留在 medical_docs/ 中，
   再运行：
       python build_vector_index.py --docs medical_docs
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

import requests


DOCS = [
    {
        "filename": "WHO_成人高血压药物治疗指南_中文版.pdf",
        "title": "成人高血压药物治疗指南",
        "publisher": "World Health Organization",
        "year": "2022",
        "lang": "zh",
        "publication_page": "https://www.who.int/zh/publications/i/item/9789240033986",
        "pdf_url": "https://iris.who.int/server/api/core/bitstreams/a93c5d03-fbb8-4f2b-9e13-5d11c4eb25c4/content",
    },
    {
        "filename": "WHO_HEARTS_健康生活方式咨询.pdf",
        "title": "HEARTS: Healthy-lifestyle counselling",
        "publisher": "World Health Organization",
        "year": "2018",
        "lang": "en",
        "publication_page": "https://www.who.int/publications/i/item/WHO-NMH-NVI-18-1",
        "pdf_url": "https://iris.who.int/server/api/core/bitstreams/0c21ba41-60db-4592-a97c-60a8b62033eb/content",
    },
    {
        "filename": "WHO_HEARTS_循证治疗方案.pdf",
        "title": "HEARTS: Evidence-based treatment protocols",
        "publisher": "World Health Organization",
        "year": "2018",
        "lang": "en",
        "publication_page": "https://www.who.int/publications/i/item/WHO-NMH-NVI-18-2",
        "pdf_url": "https://iris.who.int/server/api/core/bitstreams/21769583-367d-47f8-b1a3-4e1abc4a9639/content",
    },
]


def download_one(session: requests.Session, doc: dict, out_dir: Path) -> bool:
    out_path = out_dir / doc["filename"]

    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[SKIP] 已存在: {out_path.name}")
        return True

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        "Referer": doc["publication_page"],
    }

    print(f"\n[DOWNLOAD] {doc['title']}")
    print(f"           -> {out_path}")

    try:
        with session.get(
            doc["pdf_url"],
            headers=headers,
            timeout=60,
            stream=True,
            allow_redirects=True,
        ) as response:
            content_type = response.headers.get("content-type", "")
            print(
                f"           HTTP {response.status_code} | "
                f"{content_type}"
            )

            if response.status_code != 200:
                return False

            first = True

            with out_path.open("wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 256
                ):
                    if not chunk:
                        continue

                    if first:
                        # PDF 一般以 %PDF 开头；不是 PDF 时不要污染 medical_docs
                        if b"%PDF" not in chunk[:1024]:
                            print(
                                "[WARN] 返回内容不是 PDF，"
                                "取消保存。"
                            )
                            f.close()
                            out_path.unlink(missing_ok=True)
                            return False

                        first = False

                    f.write(chunk)

        print(
            f"[OK] {out_path.name} "
            f"({out_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return True

    except Exception as exc:
        print(f"[ERROR] {exc}")
        out_path.unlink(missing_ok=True)
        return False


def main() -> None:
    out_dir = Path("medical_docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    failed = []

    print("=" * 72)
    print("WHO Medical Docs Starter Pack")
    print("目标目录:", out_dir.resolve())
    print("=" * 72)

    for doc in DOCS:
        ok = download_one(
            session,
            doc,
            out_dir,
        )

        if not ok:
            failed.append(doc)

    print("\n" + "=" * 72)

    if not failed:
        print("全部下载成功。")
        print(
            "下一步执行："
            "python build_vector_index.py --docs medical_docs"
        )
        return

    print(
        f"{len(failed)} 个文档未能通过脚本下载。"
    )
    print(
        "WHO IRIS 有时会限制程序化下载，"
        "这不是你的 Python 环境问题。"
    )
    print(
        "下面将列出官方出版页；"
        "你可以在页面上点击 Download/PDF。"
    )

    for index, doc in enumerate(
        failed,
        start=1,
    ):
        print(
            f"{index}. {doc['title']}\n"
            f"   {doc['publication_page']}"
        )

    answer = input(
        "\n是否依次用浏览器打开这些 WHO 官方页面？[y/N]: "
    ).strip().lower()

    if answer == "y":
        for doc in failed:
            webbrowser.open(
                doc["publication_page"]
            )


if __name__ == "__main__":
    main()

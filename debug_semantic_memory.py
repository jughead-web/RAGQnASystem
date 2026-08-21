from semantic_memory import SemanticMemory


memory=SemanticMemory()


results=memory.search(
    "高血压患者使用什么药降低血压？"
)


for r in results:

    print("="*50)

    print(r.payload["name"])

    print(r.payload["text"])

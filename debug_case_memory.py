from case_memory import CaseMemory


memory = CaseMemory()


query = "65岁高血压血压160应该怎么治疗"


results = memory.search(query)


for r in results:

    print("="*50)

    print(
        r.payload["patient_id"]
    )

    print(
        r.payload["content"]
    )

    print(
        "score:",
        r.score
    )


memory.client.close()
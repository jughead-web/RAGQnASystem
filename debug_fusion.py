# -*- coding:utf-8 -*-

from context_fusion import MedicalContextFusion



fusion = MedicalContextFusion()



query = """
65岁男性，
血压160/95 mmHg，
诊断原发性高血压，
服用氨氯地平5mg每日一次，
8周后血压135/85，
是否继续当前治疗？
"""



result = fusion.retrieve(query)



print("\n")
print("="*80)

print("【指南 Evidence】")

for item in result["guidelines"]:

    print(
        "\n标题:",
        item.title
    )

    print(
        "页码:",
        item.page
    )

    print(
        "分数:",
        item.score
    )

    print(
        item.content[:300]
    )



print("\n")
print("="*80)


print("【相似病例】")

for item in result["cases"]:

    print(item)



print("\n")
print("="*80)


print("【语义知识】")

for item in result["semantic"]:

    print(item)
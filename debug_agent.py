from medical_agent import MedicalAgent


agent = MedicalAgent()


query = """
65岁男性，
血压160/95 mmHg，
诊断原发性高血压，
服用氨氯地平5mg每日一次，
8周后血压135/85mmHg，
是否需要调整治疗？
"""


answer = agent.ask(query)


print("="*80)
print(answer)


agent.close()
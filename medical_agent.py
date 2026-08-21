# -*- coding: utf-8 -*-

from context_fusion import MedicalContextFusion
import ollama


class MedicalAgent:
    """
    医疗多记忆增强 Agent

    Pipeline:

    Query
      |
      v
    Context Fusion
      |
      ├── Global Medical KB
      ├── Case Memory
      └── Semantic Memory
      |
      v
    Ollama LLM
      |
      v
    Answer
    """

    def __init__(
        self,
        model="qwen:7b"
    ):

        print("[Agent] Loading Fusion")

        self.fusion = MedicalContextFusion()

        self.model = model



    def format_guidelines(
        self,
        guidelines
    ):

        text = ""

        if not guidelines:

            return "无医学指南证据"


        for i, item in enumerate(
            guidelines,
            start=1
        ):

            text += f"""

[指南证据 {i}]

来源:
{item.title}

页码:
{item.page}

内容:
{item.content}

"""

        return text



    def format_cases(
        self,
        cases
    ):

        text = ""

        if not cases:

            return "无相似病例"


        for i, item in enumerate(
            cases,
            start=1
        ):

            payload = item.payload


            text += f"""

[病例 {i}]


患者:
{payload.get("patient_id")}


年龄:
{payload.get("age")}


诊断:
{payload.get("diagnosis")}


治疗方案:
{payload.get("medications")}


病程:
{payload.get("events")}


"""


        return text




    def format_semantic(
        self,
        semantic
    ):

        text = ""


        if not semantic:

            return "无药物知识"


        for i, item in enumerate(
            semantic,
            start=1
        ):

            payload = item.payload


            text += f"""

[医学实体 {i}]

{payload.get("text","")}

"""


        return text




    def build_context(
        self,
        result
    ):


        context = f"""

========================
【医学指南 Global KB】
========================

{self.format_guidelines(
    result.get("guidelines", [])
)}



========================
【病例记忆 Case Memory】
========================

{self.format_cases(
    result.get("cases", [])
)}



========================
【医学语义 Semantic Memory】
========================

{self.format_semantic(
    result.get("semantic", [])
)}

"""


        return context




    def build_prompt(
        self,
        query,
        context
    ):


        prompt = f"""

你是一名医学辅助问答AI。

你的回答必须严格基于下面提供的医学证据。


回答要求：

1. 医学指南中的内容必须明确标注为“指南推荐”。

2. 病例只能作为相似案例参考，
   不允许把病例治疗方案描述为指南推荐。

3. 药物知识只能说明药物类别、机制、剂量等信息，
   不允许推断患者一定应该使用该药。

4. 如果证据中没有明确答案，
   必须说明“当前证据不足”。

5. 所有关键结论必须能够对应到提供的证据。



========================

用户问题：

{query}


========================

检索证据：

{context}


========================


请按照以下格式回答：


【指南依据】

说明医学指南中的推荐。


【病例参考】

检索到相似病例：

65岁男性，
诊断：原发性高血压

初始血压：
160/95 mmHg

治疗：
氨氯地平5mg每日一次

随访：
8周后135/85mmHg

【药物信息】

说明相关药物作用、剂量或注意事项。


【综合判断】

结合以上信息回答用户问题。


【参考来源】

列出使用的证据来源。



"""

        return prompt




    def ask(
        self,
        query
    ):


        print("[Agent] Retrieving context")


        result = self.fusion.retrieve(
            query
        )


        context = self.build_context(
            result
        )


        prompt = self.build_prompt(
            query,
            context
        )


        print("[Agent] Generating answer")


        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature":0.2
            }
        )


        return response["response"]



    def close(self):

        """
        关闭Qdrant连接
        """

        try:

            if hasattr(
                self.fusion.global_kb,
                "client"
            ):

                self.fusion.global_kb.client.close()


            if hasattr(
                self.fusion.case_memory,
                "client"
            ):

                self.fusion.case_memory.client.close()


            if hasattr(
                self.fusion.semantic_memory,
                "client"
            ):

                self.fusion.semantic_memory.client.close()


        except Exception:

            pass
# -*- coding: utf-8 -*-

from typing import Dict, Any


from global_medical_kb import GlobalMedicalRetriever
from case_memory_v2 import CaseMemoryV2
from semantic_memory import SemanticMemory



class MedicalContextFusion:
    """
    医疗多记忆融合层

    Memory:
        1. Global Medical KB
           医学指南 / 循证知识

        2. Case Memory
           相似患者病例经验

        3. Semantic Memory
           药品说明 / 医学实体知识
    """


    def __init__(self):

        print("[Fusion] Loading Global Medical KB")

        self.global_kb = GlobalMedicalRetriever()



        print("[Fusion] Loading Case Memory")

        self.case_memory = CaseMemoryV2()



        print("[Fusion] Loading Semantic Memory")

        self.semantic_memory = SemanticMemory()



    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:


        result = {

            "guidelines": [],

            "cases": [],

            "semantic": []

        }



        # =========================
        # 1. 医学指南
        # =========================

        try:

            result["guidelines"] = (
                self.global_kb.search(
                    query,
                    top_k=top_k
                )
            )

        except Exception as e:

            print(
                "[Global KB Error]",
                e
            )



        # =========================
        # 2. 病例记忆
        # =========================

        try:

            result["cases"] = (
                self.case_memory.search(
                    query,
                    top_k=top_k
                )
            )

        except Exception as e:

            print(
                "[Case Memory Error]",
                e
            )



        # =========================
        # 3. 药物语义记忆
        # =========================

        try:

            result["semantic"] = (
                self.semantic_memory.search(
                    query,
                    top_k=top_k
                )
            )


        except Exception as e:

            print(
                "[Semantic Memory Error]",
                e
            )



        return result
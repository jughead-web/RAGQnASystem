# -*- coding:utf-8 -*-

import json
import uuid

from fastembed import TextEmbedding

from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)


COLLECTION_NAME = "medical_case_memory_v2"



class CaseMemoryV2:


    def __init__(self):

        self.client = QdrantClient(
            path="qdrant_storage/case_memory"
            )


        self.embed = TextEmbedding(
            model_name="intfloat/multilingual-e5-large"
        )


        self.init_collection()



    def init_collection(self):

        names = [
            c.name
            for c in self.client.get_collections().collections
        ]


        if COLLECTION_NAME not in names:

            self.client.create_collection(

                collection_name=COLLECTION_NAME,

                vectors_config=VectorParams(

                    size=1024,

                    distance=Distance.COSINE

                )

            )



    def case_to_text(self, case):


        text = f"""

患者基本信息:

性别:
{case['basic_info']['gender']}

年龄:
{case['basic_info']['age']}


疾病:

{','.join(case['diagnosis'])}


症状:

{','.join(case['symptoms'])}


检查结果:

{case['lab_results']}


药物:

{case['medications']}


病例事件:

"""


        for event in case["events"]:

            text += f"""

时间:
{event['date']}

事件:
{event['event_type']}

描述:
{event['description']}

"""


        return text



    def add_case(self, json_path):


        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            case=json.load(f)



        text=self.case_to_text(case)


        vector=list(

            self.embed.passage_embed(

                [
                    text
                ]

            )

        )[0]



        point=PointStruct(

            id=str(uuid.uuid4()),


            vector=vector.tolist(),


            payload={

                "patient_id":
                    case["patient_id"],


                "diagnosis":
                    case["diagnosis"],


                "age":
                    case["basic_info"]["age"],


                "medications":
                    case["medications"],


                "events":
                    case["events"],


                "content":
                    text,

                "memory_type":
                    "episodic"

            }

        )


        self.client.upsert(

            collection_name=COLLECTION_NAME,

            points=[
                point
            ]

        )



    def search(
        self,
        query,
        top_k=3
    ):


        vector=list(

            self.embed.query_embed(

                [
                    "query: "+query
                ]

            )

        )[0]


        result=self.client.query_points(

            collection_name=COLLECTION_NAME,

            query=vector.tolist(),

            limit=top_k,

            with_payload=True

        )


        return result.points
    def close(self):
        try:
            self.client.close()
        except:
            pass
# -*- coding:utf-8 -*-

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from fastembed import TextEmbedding


COLLECTION_NAME = "medical_case_memory"


class CaseMemory:


    def __init__(self):

        self.client = QdrantClient(
            path="qdrant_storage"
        )


        self.embed = TextEmbedding(
            model_name="intfloat/multilingual-e5-large"
        )


        self._init_collection()



    def _init_collection(self):

        collections = [
            c.name 
            for c in self.client.get_collections().collections
        ]


        if COLLECTION_NAME not in collections:

            self.client.create_collection(
                collection_name=COLLECTION_NAME,

                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )



    def add_case(
        self,
        text,
        patient_id
    ):


        vector = list(
            self.embed.passage_embed(
                [text]
            )
        )[0]


        point = PointStruct(

            id=str(uuid.uuid4()),

            vector=vector.tolist(),

            payload={

                "patient_id":patient_id,

                "type":"patient_case",

                "content":text

            }
        )


        self.client.upsert(

            collection_name=COLLECTION_NAME,

            points=[point]

        )



    def search(
            self,
            query,
            top_k=3
            ):
         vector = list(
             self.embed.query_embed(
                 [
                     "query: " + query
                     ]
                     )
                     )[0]
         result = self.client.query_points(
             collection_name=COLLECTION_NAME,
             query=vector.tolist(),
             limit=top_k,
             with_payload=True
             )
         return result.points
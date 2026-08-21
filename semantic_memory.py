# -*- coding:utf-8 -*-

import os
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from fastembed import TextEmbedding


class SemanticMemory:


    def __init__(
        self,
        data_dir="semantic_memory",
        collection_name="semantic_memory"
    ):

        self.data_dir=data_dir

        self.collection_name=collection_name


        self.embed = TextEmbedding(
            model_name="intfloat/multilingual-e5-large"
        )


        self.client = QdrantClient(
            path="qdrant_storage/semantic_memory"
            )


        self._init_collection()



    def _init_collection(self):

        collections=[
            x.name 
            for x in self.client.get_collections().collections
        ]


        if self.collection_name not in collections:

            self.client.create_collection(

                collection_name=self.collection_name,

                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )



    def build(self):

        points=[]


        for file in os.listdir(self.data_dir):

            if not file.endswith(".json"):
                continue


            path=os.path.join(
                self.data_dir,
                file
            )


            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data=json.load(f)



            text=self._json_to_text(data)


            vector=list(
                self.embed.embed(
                    [text]
                )
            )[0]


            points.append(

                PointStruct(

                    id=str(uuid.uuid4()),

                    vector=vector.tolist(),

                    payload={
                        "drug_id":data["drug_id"],
                        "name":data["name"],
                        "text":text
                    }
                )
            )



        self.client.upsert(

            collection_name=self.collection_name,

            points=points
        )


        print(
            f"Semantic Memory build success: {len(points)}"
        )




    def search(
        self,
        query,
        top_k=3
    ):


        vector=list(
            self.embed.embed(
                [query]
            )
        )[0]


        result=self.client.query_points(

            collection_name=self.collection_name,

            query=vector.tolist(),

            limit=top_k
        )


        return result.points




    def _json_to_text(self,data):

        text=""


        for k,v in data.items():

            text+=f"{k}:{v}\n"


        return text
    def close(self):
        try:
            self.client.close()
        except:
            pass
import json
from typing import List
import os
from redis_stack.redis_vector_search import RedisClient
from models.embedder import Embedder


def get_json(data: List[str]):
    final_data = []
    for json_doc in data:
        with open(json_doc, 'r', encoding="utf-8") as f:
            temp = dict(json.load(f))
            all_links = temp.keys()
            for link in all_links:
                sub_dict = temp[link]
                for question, answer in sub_dict.items():
                    final_data.append({"question": question.replace("\n", ""),
                                       "answer": answer.replace("\n", ""),
                                       "link": link})

    return final_data


if __name__ == "__main__":
    path_to_json = os.getcwd()
    data = get_json(pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json'))
    embedding_model = Embedder('distiluse-base-multilingual-cased-v1')
    redis = RedisClient(model=embedding_model)
    redis.client.flushdb()
    redis.store_new_data(data=data)
    redis.create_vector_field()
    key, value = redis.create_embeddings()
    redis.store_embeddings(keys=key, embeddings=value)

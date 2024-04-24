import redis
import numpy as np
from redis.commands.search.field import (
    NumericField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from models.embedder import Embedder
import os


class RedisClient:
    def __init__(self, HOST=os.environ.get('REDIS_HOST'), PORT=os.environ.get('REDIS_PORT')
                 , PASSWORD=os.environ.get('REDIS_PASSWORD'), model: Embedder = None):
        try:
            self.client = redis.Redis(host=HOST, port=PORT, password=PASSWORD,
                                      decode_responses=True)
            self.embedding_model = model
        except redis.RedisError:
            pass

    def store_new_data(self, data: list[dict]):
        pipeline = self.client.pipeline()
        for index, program in enumerate(data):
            redis_key = index
            pipeline.json().set(redis_key, "$", program)
        pipeline.execute()
        pipeline.reset()

    def create_vector_field(self):
        try:
            self.client.ft("idx:docs").info()
            print('Vector_Field_is_already_created')
        except redis.exceptions:
            vector_dim = len(self.embedding_model.get_embedding('vec'))
            schema = (
                NumericField("$.id", as_name="id"),
                TextField("$.question", as_name="question"),
                TextField("$.answer", as_name="answer"),
                VectorField(
                    "$.questions_embeddings",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="question_vector",
                ),
            )
            definition = IndexDefinition(prefix=["document:"], index_type=IndexType.JSON)
            self.client.ft("idx:docs").create_index(
                fields=schema, definition=definition
            )

    def create_embeddings(self):
        keys = sorted(self.client.keys('document:*'))
        question = self.client.json().mget(keys, '$.question')
        question = [item for sublist in question for item in sublist]
        embeddings = self.embedding_model.get_embedding(question)
        return keys, embeddings

    def store_embeddings(self, keys: list, embeddings: list):
        pipeline = self.client.pipeline()
        for key, embedding in zip(keys, embeddings):
            pipeline.json().set(key, "$.questions_embeddings", embedding)
        pipeline.execute()

    def search_query(self, k: int, user_text: str):
        query = (
            Query(f'(*)=>[KNN {k} @question_vector $query_vector AS vector_score]')
            .sort_by('vector_score')
            .return_fields('vector_score', 'id')
            .dialect(2)
        )

        encoded_query = self.embedding_model.get_embedding(user_text)
        result_list = self.client.ft("idx:docs").search(query, {
            'query_vector': np.array(encoded_query, dtype=np.float32).tobytes()}).docs
        res = []
        for index in result_list:
            res.append(index.id)
        return res

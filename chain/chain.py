from langchain.llms import HuggingFacePipeline
from models.LLM import LanguageModel
from redis_stack.redis_vector_search import RedisClient
from langchain.chains import RetrievalQA


class Chain:
    def __init__(self, llm: LanguageModel, vector_db: RedisClient):
        self.llm = HuggingFacePipeline(pipeline=llm.generate_text())
        self.vector_space = vector_db

    def execute_search(self, k: int, query: str):
        top_matches = self.vector_space.search_query(k=k, user_text=query)
        return top_matches

    def reranking(self):
        pass

    def get_answer(self, k: int, query: str):
        retrieval_qa = RetrievalQA(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vector_space,
            retrieve_method=self.vector_space.search_query,
            response_processing_function=lambda x: x
        )
        return retrieval_qa.answer(query=query, k=k)

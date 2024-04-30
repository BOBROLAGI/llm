from redis_stack.redis_vector_search import RedisClient
from models.LLM import LanguageModel


class ChatAgent:
    def __init__(self, session_id: int, language_model: LanguageModel, redis_client: RedisClient):
        self.session_id = session_id
        self.redis_client = redis_client
        self.language_model = language_model

    def retrieve_documents(self, query: str, k: int):
        document_ids = self.redis_client.search_query(k, query)
        return document_ids

    def get_context_from_ids(self, document_ids: list):
        context = ""
        links = []
        for doc_id in document_ids:
            document_text = self.redis_client.client.json().get(doc_id)['answer']
            document_link = self.redis_client.client.json().get(doc_id)['link']
            context += document_text
            links.append(document_link)
        return context, links

    def generate_response(self, context: str, question: str, links: list):
        response = self.language_model.model.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": self.language_model.prompt_template.format(context=context, links=links)},
                {"role": "user",
                 "content": question}],

            temperature=0.1,
        )

        return response['choices'][0]['message']['content']

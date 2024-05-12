from redis_stack.redis_vector_search import RedisClient
from models.LLM import LanguageModel
from typing import List


class ChatAgent:
    def __init__(self, session_id: str, language_model: LanguageModel, redis_client: RedisClient):
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

            temperature=0.3,
            max_tokens=2000,
        )

        return response['choices'][0]['message']['content']

    def generate_summarization(self, text: str, k: str = "3-5", header='заголовка'):
        response = self.language_model.model.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": self.language_model.summarization_template.format(user_prompt=text, k=k, header=header)}],
            temperature=0.3,
            max_tokens=10,
        )
        return response['choices'][0]['message']['content']

    def generate_response_with_history(self, context: str, question: str, links: list, history: List[str]):
        messages = ""
        for message in history:
            messages += message
        history = self.generate_summarization(messages, k="20-30", header="сводки")
        response = self.language_model.model.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": self.language_model.prompt_template.format(context=context, links=links) + history},
                {"role": "user",
                 "content": question}],
            temperature=0.3,
            max_tokens=3000,
        )

        return response['choices'][0]['message']['content']

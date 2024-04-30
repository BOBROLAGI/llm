import asyncio
from models.LLM import LanguageModel
from models.embedder import Embedder
from redis_stack.redis_vector_search import RedisClient
from rag.agent import ChatAgent
from rag.chat_history import ChatHistory
from data.process_output import process_output


async def handle_user_message(message: str, session_id: int, agent: ChatAgent, chat_history: ChatHistory):
    chat_history.add_message_to_history(message, session_id)
    documents = agent.retrieve_documents(query=message, k=2)
    context, links = agent.get_context_from_ids(document_ids=documents)
    bot_response = agent.generate_response(context=context, question=message, links=links)
    chat_history.add_message_to_history(bot_response, session_id)
    return bot_response, links


async def start_chat(user_messages: list, session_id: int):
    embedding_model = Embedder('distiluse-base-multilingual-cased-v1')
    language_model = LanguageModel(model_name='artifacts/model-q8_0.gguf')
    vector_db = RedisClient(model=embedding_model)
    agent = ChatAgent(session_id=session_id, language_model=language_model, redis_client=vector_db)
    chat_history = ChatHistory()
    for message in user_messages:
        bot_response, links = await handle_user_message(message, session_id, agent, chat_history)
        print(f"Bot: {bot_response}, Полезные ссылки: {links}")


async def main():
    user_messages = ["Как оформить карту?"]

    session_id = 1

    await start_chat(user_messages, session_id)


asyncio.run(main())

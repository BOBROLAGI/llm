from pymongo import MongoClient
from dotenv import load_dotenv
import os
import datetime
from rag.agent import ChatAgent
load_dotenv()


class ChatHistory:
    def __init__(self):
        self.uri = (f"mongodb+srv://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASSWORD')}@sberml.jbtvzn5"
                    f".mongodb.net/")
        self.client = MongoClient(self.uri)

        self.db = self.client['chat_history']
        self.collection = self.db['chats']

    def add_message_to_history(self, sender: str, message: str, chat_id: str,  agent: ChatAgent):
        message_data = {"sender": sender, "text": message, "timestamp": datetime.datetime.utcnow()}
        chat = self.collection.find_one({"chat_id": chat_id})
        if chat:
            self.collection.update_one({"chat_id": chat_id}, {"$push": {"messages": message_data}})
        else:
            title = agent.generate_summarization(message)
            chat_data = {"chat_id": chat_id, "title": title, "messages": [message_data], "user_id": sender}
            self.collection.insert_one(chat_data)

    def delete_message_from_history(self, session_id: str):
        self.collection.delete_one({"chat_id": session_id})

    def get_chat_history_by_user(self, user_id: str):
        return self.collection.find({"user_id": user_id})

    def get_chat_history_by_chat_id(self, chat_id: str):
        return self.collection.find_one({"chat_id": chat_id})

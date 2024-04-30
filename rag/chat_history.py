from pymongo import MongoClient
from dotenv import load_dotenv
import os
import datetime
load_dotenv()


class ChatHistory:
    def __init__(self):
        self.uri = f"mongodb+srv://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASSWORD')}@sberml.jbtvzn5.mongodb.net/"
        self.client = MongoClient(self.uri)

        self.db = self.client['chat_history']
        self.collection = self.db['messages']

    def add_message_to_history(self, message: str, session_id: int):
        message_data = {"text": message, "timestamp": datetime.datetime.utcnow()}
        chat = self.collection.find_one({"session_id": session_id})
        if chat:
            self.collection.update_one({"session_id": session_id}, {"$push": {"messages": message_data}})
        else:
            self.collection.insert_one({"session_id": session_id, "messages": [message_data]})

    def delete_message_from_history(self, session_id: int):
        self.collection.delete_one({"session_id": session_id})



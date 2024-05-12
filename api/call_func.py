import asyncio
from fastapi.middleware.cors import CORSMiddleware
from models.LLM import LanguageModel
from models.embedder import Embedder
from redis_stack.redis_vector_search import RedisClient
from rag.agent import ChatAgent
from rag.chat_history import ChatHistory
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any
from fastapi.encoders import jsonable_encoder
from datetime import datetime
import random
from dataclasses import dataclass
import uuid
import bson
from bson.json_util import dumps
import json

app = FastAPI()
origins = [""]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = Embedder('distiluse-base-multilingual-cased-v1')
language_model = LanguageModel(model_name='../artifacts/model-q8_0.gguf')
vector_db = RedisClient(model=embedding_model)
chat_history = ChatHistory()


class Message(BaseModel):
    sender: str
    text: str
    timestamp: datetime


class Chat(BaseModel):
    chat_id: str
    title: str
    messages: List[Message]
    links: List[str]


class ChatCreationRequest(BaseModel):
    message: str
    user_id: str


class SendMessageRequest(BaseModel):
    text: str
    chat_id: str
    user_id: str


class SendMessageResponse(BaseModel):
    text: str
    links: List[str]
    chat_id: str


@app.post("/chats")
async def create_chat(chat_request: ChatCreationRequest) -> SendMessageResponse:
    chat_id = str(uuid.uuid4())
    agent = ChatAgent(session_id=chat_id, language_model=language_model, redis_client=vector_db)
    chat_history.add_message_to_history(chat_id=chat_id, message=chat_request.message,
                                        sender=f"{chat_request.user_id}", agent=agent)
    documents = agent.retrieve_documents(query=chat_request.message, k=2)
    context, links = agent.get_context_from_ids(document_ids=documents)
    bot_response = agent.generate_response(context=context, question=chat_request.message, links=links)
    chat_history.add_message_to_history("bot", bot_response, chat_id, agent)
    return SendMessageResponse(text=bot_response, links=links, chat_id=chat_id)


@app.post("/message")
async def handle_message(request: SendMessageRequest) -> SendMessageResponse:
    agent = ChatAgent(session_id=request.chat_id, language_model=language_model, redis_client=vector_db)
    documents = agent.retrieve_documents(query=request.text, k=2)
    context, links = agent.get_context_from_ids(document_ids=documents)
    history = chat_history.get_chat_history_by_chat_id(request.chat_id)["messages"]
    history = [message["text"] for message in history]
    bot_response = agent.generate_response_with_history(context=context, question=request.text, links=links,
                                                        history=history)
    chat_history.add_message_to_history(sender=f"{request.user_id}", message=request.text,
                                        chat_id=request.chat_id, agent=agent)
    chat_history.add_message_to_history("bot", bot_response, request.chat_id, agent)
    return SendMessageResponse(text=bot_response, links=links, chat_id=request.chat_id)


@app.get("/user/{user_id}")
async def get_all_user_chats(user_id: str) -> Any:
    chats = chat_history.get_chat_history_by_user(user_id)
    return JSONResponse(content=json.loads(dumps(chats)))


@app.get("/chat/{chat_id}")
async def get_chat_by_id(chat_id: str) -> Any:
    chat = chat_history.get_chat_history_by_chat_id(chat_id)
    messages = []
    for message in chat["messages"]:
        messages.append(Message(sender=message["sender"], text=message["text"],
                                timestamp=message["timestamp"].isoformat()))
    return {
        "chat_id": chat["chat_id"],
        "messages": messages,
        "title": chat["title"]}

from langchain.agents import load_tools, Tool
from langchain.tools import StructuredTool
import requests
from pydantic import BaseModel, Field


class SingleInputToolSchema(BaseModel):
    query: str = Field(..., min_length=1)


def send_telegram_message(message):
    url = "https://api.telegram.org/bot1732866192:AAEp_3tXo994Yy8eUQm2xjUpgU-Pg4XvIi0/sendMessage"
    data = {"chat_id": "1719079956", "text": message}

    response = requests.post(url, data=data)
    return "Message sent!"


def _get_llm_math(llm):
    return load_tools(["llm-math"], llm=llm)[0]


def _get_retriever(retriever, name, description):
    return StructuredTool(
        name=name,
        description=description,
        func=retriever.get_relevant_documents,
        args_schema=SingleInputToolSchema,
    )


def _get_telegram_tool(description=None):
    return Tool(
        name="send_telegram_message",
        description=description or "Used to submit user info (name, email, phone number) and send any other kind of messages or updates to the owner of the jewelry store.",
        func=send_telegram_message,
    )


tool_dict = {
    "calculator": _get_llm_math,
    "retriever": _get_retriever,
    "telegram": _get_telegram_tool,
}


def get_tool(tool_name):
    return tool_dict[tool_name]

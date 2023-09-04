from langchain.agents import load_tools, Tool
from langchain.tools import StructuredTool
import requests
from pydantic import BaseModel, Field


class SingleInputToolSchema(BaseModel):
    query: str = Field(..., min_length=1)

class SendToMakeArgsSchema(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)


def send_dummy_image_url(query):
    return "return this image id object to the user as it is without any modification: ImageID(3324)"


def send_telegram_message(query):
    url = "https://api.telegram.org/bot1732866192:AAEp_3tXo994Yy8eUQm2xjUpgU-Pg4XvIi0/sendMessage"
    data = {"chat_id": "1719079956", "text": query}

    response = requests.post(url, data=data)
    return "Message sent!"

def send_to_make(name, email):
    url = 'https://hook.eu2.make.com/60u3iqfbpqbsealpjl61qvu1p5nc2tpg'

    json = {
        "name": name,
        "email": email,
    }

    response = requests.post(url, json=json)
    if response.text == "Accepted":
        return "Data saved successfully!"

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
    return StructuredTool(
        name="send_telegram_message",
        description=description or "Used to submit user info (name, email, phone number) and send any other kind of messages or updates to the owner of the jewelry store.",
        func=send_telegram_message,
        args_schema=SingleInputToolSchema,
    )


def _get_dummy_image_tool():
    return StructuredTool(
        name="image_getter_tool",
        description="Use this tool to look for image id of an image using its description",
        func=send_dummy_image_url,
        args_schema=SingleInputToolSchema,
    )

def _get_make_tool():
    return StructuredTool(
        name="save_user_data",
        description="Use this tool to save user data (name, email) to the excel sheet",
        func=send_to_make,
        args_schema=SendToMakeArgsSchema,
    )


tool_dict = {
    "calculator": _get_llm_math,
    "retriever": _get_retriever,
    "telegram": _get_telegram_tool,
    "image": _get_dummy_image_tool,
    "make": _get_make_tool,
}


def get_tool(tool_name):
    return tool_dict[tool_name]

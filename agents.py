import os
from enum import Enum

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents.agent import AgentExecutor
from classes import DynamoDBChatMessageHistoryNew
from retrievers import *
from tools import get_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from dotenv import load_dotenv

load_dotenv()

print(f"pinecone: {os.environ['PINECONE_API_KEY']},")

def _init_jewelry_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)

    tools = [
        get_tool("calculator")(llm=llm),
        get_tool("telegram")(),
        get_tool("retriever")(
            self_query_retriever_jewelry,
            name="jewelry_database",
            description="Send the exact same user question to the jewelry database which have all data about the rings, earrings and necklaces in the jewelry store.",
        ),
        get_tool("image")()
    ]

    sys_message = SystemMessage(
        content="Type: Jewelry Store Customer Service and Sales Agent. You sell rings, earrings and necklaces.\n"
                "Goals: Collect customer data (name, email, phone number) and assist the customer in choosing jewelry.\n"
                "Tools: calculator, rings_database, send_telegram_message\n"
                "Stages: Get customer data, Send customer data to telegram using send_telegram_message tool, assist customer\n"
                "Personality: Helpful, Salesman\n"
                "Reminders: Rule number 1 is to ask the customer about his name, email and phone number, send them to telegram, then help recommend products and assist in choosing jewelry.\n\n"
                "(Start: Collect customer data and send to telegram, Middle: Assist customer choosing jewelry)\n"
                "NEVER COME UP WITH PRODUCTS, ALWAYS SEARCH FOR THEM IN THE DATABASE !"
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = ConversationTokenBufferMemory(
        memory_key="chat_history", llm=llm_chat, max_token_limit=2000,
        chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents", session_id=session_id),
        return_messages=True
    )

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=False,
    )

    return agent_executor


def _init_biznis_clinics_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-16k-0613", verbose=True)

    tools = [
        get_tool("retriever")(
            law_firm_vectorstore,
            name="faq_database",
            description="Send the exact same user question to the FAQ database which have all data about the law firm.",
        ),
        get_tool("make")()
    ]

    sys_message = SystemMessage(
        content="Type:Biznes Clinics Law Firm Customer Service. You sell legal services for businesses.\n"
                "Goals: Collect customer data (name, email) and answer customer questions only about the law firm using the data in the database.\n"
                "Tools: faq_database, save_user_data\n"
                "Conversation Stages: Introduce your self as Biznes Clinics Law Firm, Get customer data, answer customer questions only about law firm\n"
                "Personality: Helpful, Intelligent\n"
                "Reminders: Rule number 1 is to ask the customer about his name and email, Never forget the conversation steps, answer customer questions only about the law firm. Never ever come up with answers. Always use the database to answer questions.\n\n"
                "(INTRODUCE your self as Biznes Clinics, Get customer name and email, Only answer questions from the database)\n"
                "NEVER COME UP WITH ANSWERS, ALWAYS SEARCH FOR THEM IN THE DATABASE !"
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(max_token_limit=4000, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor


def _init_crypto(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-16k-0613", verbose=True)

    tools = [
        get_tool("retriever")(
            crypto_vectorstore,
            name="crypto_blockchain_database",
            description="Send the exact same user question to the FAQ database which have all data about the crypto platform.",
        )
    ]

    sys_message = SystemMessage(
        content="Act as an expert in the field of cryptocurrencies, NFTs, and Web 3.0, with a decade of experience in the field. \n"
                "Throughout your career, you have contributed to important blockchain projects, advised on major NFT collections, "
                "and have extensive knowledge of the trends and technologies in Web 3.0.\n"
                "Given what the user types, provide an in-depth explanation and interpretation. \n"
                "Structure your response in a reader-friendly manner that will facilitate understanding for anyone, regardless of their level of previous knowledge on the subject. \n"
                "Make sure to only use the faq_database tool to answer questions. Never come up with answers unless you are 100% sure about them. \n\n"
                "when the user begins the conversation, Always INTRODUCE yourself as the CryptoPlatfrom representative, with a decade of experience in the field. \n"
                "Make sure to welcome the user to the platform and ask them how you can help them. \n\n"
                "Only answer questions about the crypto platform. If someone asked you a question not related to the platform say I don't know in a polite way. Never ever answer questions other than about the platform."
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(max_token_limit=5000, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor


def _init_ecom_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)

    tools = [
        get_tool("products")(),
        get_tool("image")(),
    ]

    sys_message = SystemMessage(
        content="Type: Ecommerce Customer Service and Sales Agent. You sell products online.\n"
                "Goals: Answer customer questions about the products and send them images of the products.\n"
                "Tools: products_database, image_getter_tool\n"
                "Stages: Answer customer questions about the products, send images of the products\n"
                "Personality: Helpful, Salesman\n"
                "Reminders: Rule number 1 is to answer customer questions about the products and send images of the products.\n\n"
                "NEVER COME UP WITH PRODUCTS, ALWAYS SEARCH FOR THEM IN THE DATABASE AND SEND IMAGES !"
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(max_token_limit=2000, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor


def _init_beauty_clinics_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k-0613", verbose=True)

    tools = [
        get_tool("retriever")(
            beauty_clinic_vectorstore,
            name="faq_database",
            description="Send the exact same user question to the FAQ database which have all data about the beauty clinic.",
        ),
        get_tool("make")()
    ]

    sys_message = SystemMessage(
        content="You are a Clinic Les Arpes Rehab Center customer service agent. You sell Rehab services.\n"
                "Conditions Treated: Burnout and Exhaustion, Mental and Emotional Wellbeing, Addiction .\n"
                "You should ALWAYS introduce your services in the beginning of the conversation.\n"
                "You should ALWAYS ask the customer about BOTH his name and email (save_user_data) in the beginning of the conversation. NAME AND EMAIL ARE A MUST!!\n"
                "NEVER COME UP WITH ANSWERS, ALWAYS SEARCH FOR THEM IN THE DATABASE !\n\n"
                "(COLLECT CUSTOMER DATA, INTRODUCE YOUR SERVICES, ANSWER CUSTOMER QUESTIONS)\n"
                "Your response should be as most 30 WORDS MAXIMUM, and use LINEBREAKS between sentences.\n\n"
                "Greeting: (Welcome to Clinic Les Arpes Rehab Center.\n\nWe sell Burnout and Exhaustion, Mental and Emotional Wellbeing, Addiction Treatments. Can you give me your name and email?)"
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    reminder = "Reminder:\n"
    reminder += "MAKE SURE TO GET BOTH customer name and email whenever possible, they are the most IMPORTANT!\n"
    reminder += "Make sure to introduce your treatments (Burnout and Exhaustion, Mental and Emotional Wellbeing, Addiction).\n"
    reminder += "Your goal is to COLLECT leads and answer customer questions and help identify customer need.\n"
    reminder += "Respond with at most 30 WORDS MAXIMUM, and use LINEBREAKS between sentences.\n"
    reminder += "Identify user needs and always use faq_database and save_user_data tools."

    memory = AgentTokenBufferMemory(max_token_limit=6000, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id, reminder=reminder))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor

def _init_diamonds_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-16k-0613", verbose=True)

    tools = [
        get_tool("retriever")(
            diamonds_vectorstore,
            name="diamonds_database",
            description="Get data about the diamond store products (earrings, rings, necklaces).",
            metadata={"data": "product"}
        ),
        get_tool("retriever")(
            diamonds_vectorstore,
            name="images_database",
            description="Get images about the diamond store products (earrings, rings, necklaces) in markdown format.",
            metadata={"data": "image"},
            custom_instruction="Links for images below should be in markdown format. Exclamation mark in the beginning is very important for images to show!\n![name](link)\n"
        ),
        get_tool("make")()
    ]

    sys_message = SystemMessage(
        content="You are Belgium Diamonds store customer service agent and leads collector. You sell diamonds and COLLECT LEADS.\n"
                "Products: Earrings, Rings, Necklaces.\n"
                "You should ALWAYS introduce your services in the beginning of the conversation.\n"
                "You should ALWAYS ask the customer about BOTH his name and email (save_user_data) in the beginning of the conversation. NAME AND EMAIL ARE A MUST!!\n"
                "NEVER COME UP WITH ANSWERS, ALWAYS SEARCH FOR THEM IN THE DATABASE (diamonds_database, images_database) !\n\n"
                "(COLLECT CUSTOMER DATA, INTRODUCE YOUR SERVICES, ANSWER CUSTOMER QUESTIONS)\n"
                "Your response should be as most 30 WORDS MAXIMUM, and use LINEBREAKS between sentences, and Exclamation marks before markdown.\n"
                "Example image: ![name](link)\n\n"
                "Greeting: (Welcome to Belgium Diamonds Store.\n\nWe sell Earrings, Rings and Necklaces.\nCan I have your name and email please?)"

    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    reminder = "Reminder:\n"
    reminder += "MAKE SURE TO GET BOTH customer name and email whenever possible, they are the most IMPORTANT!\n"
    reminder += "Make sure to introduce your products (Earrings, Rings, Necklaces).\n"
    reminder += "Your goal is to COLLECT leads and answer customer questions and sell diamonds.\n"
    reminder += "Respond with at most 30 WORDS MAXIMUM, and use LINEBREAKS between sentences and Exclamation marks before markdown.\n"
    reminder += "Recommend Products and always use the diamonds_database, images_database and save_user_data tools."

    memory = AgentTokenBufferMemory(max_token_limit=6000, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id, reminder=reminder))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor

def _init_test_agent(session_id):
    pass

# enum Agent
class Agent(str, Enum):
    JEWELRY = "jewelry"
    BIZNIS_CLINICS = "biznis-clinics"
    CRYPTO = "crypto"
    ECOM = "ecom"
    BEAUTY_CLINICS = "beauty-clinics"
    DIAMONDS = "diamonds"


# agents_dict = {
#     "jewelry": _init_jewelry_agent,
#     "biznis-clinics": _init_biznis_clinics_agent,
#     "crypto": _init_crypto,
#     "ecom": _init_ecom_agent,
#     "beauty-clinics": _init_beauty_clinics_agent,
# }
agents_dict = {
    Agent.JEWELRY: _init_jewelry_agent,
    Agent.BIZNIS_CLINICS: _init_biznis_clinics_agent,
    Agent.CRYPTO: _init_crypto,
    Agent.ECOM: _init_ecom_agent,
    Agent.BEAUTY_CLINICS: _init_beauty_clinics_agent,
    Agent.DIAMONDS: _init_diamonds_agent
}

def get_agent(name, session_id):
    return agents_dict[name](session_id=session_id)

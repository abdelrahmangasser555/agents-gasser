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

def _init_jewelry_agent(session_id):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_chat = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613", verbose=True)

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
                "Tools: Calculator, rings_database, send_telegram_message\n"
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
            law_firm_retriever,
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

    memory = AgentTokenBufferMemory(max_token_limit=4000, memory_key="chat_history", llm=llm_chat,chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents", session_id=session_id))

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
    llm_chat = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-16k-0613", verbose=True)

    tools = [
        get_tool("retriever")(
            law_firm_retriever,
            name="faq_database",
            description="Send the exact same user question to the FAQ database which have all data about the law firm.",
        ),
        get_tool("make")()
    ]

    sys_message = SystemMessage(
        content=""
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

agents_dict = {
    "jewelry": _init_jewelry_agent,
    "biznis-clinics": _init_biznis_clinics_agent
}


def get_agent(name, session_id):
    return agents_dict[name](session_id=session_id)

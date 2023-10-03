from agents import get_agent, Agent
import json


class AgentLambdaRunner:
    def __init__(self, agent_type: Agent):
        self.agent_type = agent_type

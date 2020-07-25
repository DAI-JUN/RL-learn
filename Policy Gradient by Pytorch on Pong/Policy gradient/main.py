from environment import Environment
from agent_pg import Agent_PG

env = Environment('Pong-v0')

agent = Agent_PG(env)
agent.train()

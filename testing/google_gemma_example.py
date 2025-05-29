from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool, ToolCallingAgent
from transformers import AutoTokenizer


if __name__=="__main__":

    model = TransformersModel(model_id="google/gemma-3-1b-it",
                              device_map="cuda"
                              )
    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)

    question1 = "What is the temperature in London now?"
    agent.run(question1)

    question2 = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    agent.run(question2)



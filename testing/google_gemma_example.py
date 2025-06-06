import math

from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool, ToolCallingAgent, FinalAnswerTool, tool
from transformers import AutoTokenizer
import math

@tool
def calculate_sigmoid(x: float) -> float:
    """
    This tools calculates the sigmoid function of a number

    Args:
        x: a number

    """
    return 1/(1+math.exp(-x))

if __name__=="__main__":


    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                              device_map="cuda", # options: cpu, cuda:0 or 'auto', 'balanced', 'balanced_low_0', 'sequential'
                              # max_new_tokens=1024
                              )
    # models tried
    # google/codegemma-1.1-2b-GGUF
    # microsoft/Phi-3-mini-128k-instruct
    # unsloth/codegemma-7b-it
    # google/gemma-3-1b-it
    # mistralai/Mistral-7B-Instruct-v0.2
    # gpt2
    # BERT
    # Mistral-7B-v0.1

    # agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    # does not work with gemma-3-1b-it. Requires a model that is trained for code generation

    # agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)
    # question = "Show me the top 5 news articles headlines in the UK."
    # agent.run(question)
    # question1 = "What is the temperature in London now?"
    # agent.run(question1)
    # question2 = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    # agent.run(question2)

    # agent =  ToolCallingAgent(tools=[calculate_sigmoid], model=model)
    # agent.run("Calculate the sigmoid of 1")
    # does not run the code with google/gemma-3-1b-it

    agent = CodeAgent(tools=[calculate_sigmoid], model=model)
    agent.run("Calculate the sigmoid of a")




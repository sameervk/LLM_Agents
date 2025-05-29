from smolagents import TransformersModel, CodeAgent, DuckDuckGoSearchTool, ToolCallingAgent
from transformers import AutoTokenizer


if __name__=="__main__":

    model = TransformersModel(model_id="mistralai/Mistral-7B-Instruct-v0.2",
                              device_map="cpu"
                              )
    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)

    agent.run("What is the temperature in London?")

    # Does not work with Mistral-7B-v0.1. Same issue with GPT2 and BERT.
    # The error is
    # """
    # raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e
    # smolagents.utils.AgentGenerationError: Error in generating model output:
    # Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
    # """
    # Apparently, the reason is that "these models are not designed for structured chat interactions causing request failures due to unsupported formatting"

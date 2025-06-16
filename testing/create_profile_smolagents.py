from smolagents import CodeAgent, ToolCallingAgent, TransformersModel, Tool, tool
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from pathlib import Path


class ProfileRetrieverVersion0(Tool):
    name = "country_profile_retriever"
    description = "Uses semantic search to retrieve and summarise profile information about a country given a scenario"
    inputs = {
        "query":{
            "type":"string",
            "description": "The name of the country for which a summary must be formed from the given scenario."
        }
    }
    output_type = "string"

    def __init__(self, path_to_scenario):

        super().__init__()
        # self.retriever = BM25Retriever.from_documents(scenario, k=10)
        self.path_to_scenario = path_to_scenario


    def forward(self, query: str):

        assert isinstance(query, str), "The search query must be a string and must be a country name"

        with open(self.path_to_scenario, "r", encoding="utf8") as file:
            scenario = file.read()


        return f"Participant: {query}\n\n Scenario:{scenario}"

@tool
def load_scenario(country: str) -> str:
    """
    This tool loads the scenario using which information about the country must be extracted and summarised

    Args:
        country: country for which the information must be extracted

    """
    scenario_path = Path.cwd().parent.joinpath("data/scenarios/arnhemdreijenseweg.md")
    with open(scenario_path, "r", encoding="utf8") as file:
        scenario = file.read()

    return f"Participant: {country}\n\n Scenario:{scenario}"

if __name__=="__main__":

    scenario_path = Path.cwd().parent.joinpath("data/scenarios/arnhemdreijenseweg.md")
    scenario_loader_tool = ProfileRetrieverVersion0(scenario_path)
    # print(scenario_loader_tool.forward("Britain"))

    model = TransformersModel(model_id="google/gemma-3-1b-it",
                              device_map="cuda",
                              max_new_tokens=1024
                              )

    agent = ToolCallingAgent(tools=[load_scenario], model=model)

    agent.run("Summarise information about Britain from the scenario")

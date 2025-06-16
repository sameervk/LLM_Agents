"""

v2: Using llamaindex ReAct Agent with tools for profile generation.

"""

import os
from pathlib import Path
import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb.errors import NotFoundError
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, FunctionAgent, BaseWorkflowAgent
import asyncio

from rnd.llama_index.utils.tools import save_response


async def run_agent(agent: AgentWorkflow | BaseWorkflowAgent, user_message: str):
    return await agent.run(user_msg=user_msg)


if __name__=="__main__":


    env_loaded = dotenv.load_dotenv(Path.cwd().parent.parent.joinpath(".env"))
    if not env_loaded:
        raise FileNotFoundError("Environment file with OpenAI API key not found.")
    OA_TOKEN = os.getenv("OA_TOKEN")

    scenario_name = "arnhemdreijenseweg"


    db = chromadb.PersistentClient(path=str(Path.cwd().parent.parent.joinpath("chroma_db")))

    try:
        chroma_collection = db.get_collection(scenario_name)
    except NotFoundError as err:
        err.add_note("\n Use `ingest_data` function to create a collection")
        raise err

    vector_store =  ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=OA_TOKEN)

    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="scenario_retrieval",
        description="Use this tool for retrieving information about the scenario to help answer the query"
    )

    save_response_tool = FunctionTool.from_defaults(
        save_response,
        name="save_output",
        description="Use this tool to save your response to a file"
    )

    save_response_agent = FunctionAgent(
        name="SaveOutputAgent",
        description="Use this tool to save your response to a file",
        system_prompt="You are an AI assistant that will use the tool provided to save the response from the BattleScenarioAgent to a user query to a file",
        tools=[save_response],
        llm=llm
    )

    query_agent_workflow = ReActAgent(
        name="BattleScenarioAgent",
        description="Looks up information about battle scenario",
        system_prompt="You are a military expert in assessing the strength of armed forces and you have access to a database containing a battle scenario. "
                      "Once you have the answer ready, you should hand off control to SaveOutputAgent so that your answer is saved to a file.",
        tools=[query_engine_tool],# save_response_tool],
        llm=llm,
        can_handoff_to=["SaveOutputAgent"]
    )

    agent = AgentWorkflow(
        agents=[query_agent_workflow, save_response_agent],
        root_agent="BattleScenarioAgent"
    )
    ## LLM is not saving the response

    query_agent = ReActAgent(
        name="BattleScenarioAgent",
        description="Looks up information about battle scenario",
        system_prompt="You are a military expert in assessing the strength of armed forces and you have access to a database containing a battle scenario.",
        tools=[query_engine_tool, save_response_tool],
        llm=llm
    )

    user_msg = "Assess and evaluate the scenario and return the detailed military capability, in numbers, of the British armed forces. Save the response using the tool provided."

    response = asyncio.run(run_agent(agent=query_agent, user_message=user_msg))

    print(response)

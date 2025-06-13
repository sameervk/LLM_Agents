"""
v1: using Function Calling Agent
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
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream
import asyncio

from testing.utils.tools import save_response


async def run_agent(agent: AgentWorkflow, user_message: str):
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

    # system_prompt = """
    #     You are an expert in assessing the strength of armed forces. Using the tool provided to access the scenario, you need to assess and return
    #     the military capability in numbers of the British armed forces.
    #     """

    agent = AgentWorkflow.from_tools_or_functions([query_engine_tool, save_response_tool],
                                                  llm=llm,
                                                  system_prompt="You are a military expert in assessing the strength of armed forces and you have access to a database containing a battle scenario."
                                                  )

    user_msg = "Assess and evaluate the scenario and return the detailed military capability, in numbers, of the German armed forces. Save your response using the tool provided."

    # response = asyncio.run(run_agent(agent=agent, user_message=user_msg))
    #
    # print(response)

    async def stream_output():
        handler = agent.run(user_msg=user_msg)

        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)

    asyncio.run(stream_output())

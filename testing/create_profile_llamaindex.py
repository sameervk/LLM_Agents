import os

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel
from smolagents import TransformersModel

import asyncio
import dotenv

dotenv.load_dotenv("../.env")
HF_TOKEN=os.getenv("HF_TOKEN_INFERENCE")

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


async def agent_func(query: str):
    # initialize llm
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=HF_TOKEN)
    # # llm = AutoModel.from_pretrained("google/gemma-3-1b-it")
    # llm = TransformersModel(model_id="google/gemma-3-1b-it",
    #                           device_map="cuda",
    #                           max_new_tokens=1024
    #                           )

    # initialize agent
    agent = AgentWorkflow.from_tools_or_functions(
        [FunctionTool.from_defaults(multiply)],
        llm=llm
    )

    response = await agent.run(query)

    return response


if __name__=="__main__":

    # response = asyncio.run(agent_func("What is 2 times 2?"))
    # print(response)

    reader = SimpleDirectoryReader(input_dir="../data/scenarios")
    documents = reader.load_data()

    db = chromadb.PersistentClient(path="../data/scenarios_db")
    chroma_collection = db.get_or_create_collection("scenarios")
    vector_store =  ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=50, chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store
    )

    nodes = pipeline.run(documents=documents)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=HF_TOKEN)
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    response = query_engine.query("Generate a profile on Britain")
    print(response)

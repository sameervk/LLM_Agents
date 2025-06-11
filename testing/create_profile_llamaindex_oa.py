import os
from openai import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import dotenv




if __name__=="__main__":


    dotenv.load_dotenv("../.env")
    OA_TOKEN = os.getenv("OA_TOKEN")

    reader = SimpleDirectoryReader(input_dir="../data/scenarios")
    documents = reader.load_data()

    db = chromadb.PersistentClient(path="../data/scenarios_db_oa")
    chroma_collection = db.get_or_create_collection("scenarios")
    vector_store =  ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=8192, chunk_overlap=0),
            OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN),
        ],
        vector_store=vector_store
    )

    nodes = pipeline.run(documents=documents)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=HF_TOKEN)
    llm = OpenAI(model="gpt-4o-mini", api_key=OA_TOKEN)

    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    # response = query_engine.query("What information can you extract about Britain's military strength")
    # print(response)

    system_prompt = """
    You are an expert in assessing the strength of armed forces. From this scenario, you need to assess the military capability in numbers of the German armed forces.
    Return the output as JSON response in the following format. Do not include any json headers in the response
    
    military_capability :
    """
    response = query_engine.query(system_prompt)
    print(response)


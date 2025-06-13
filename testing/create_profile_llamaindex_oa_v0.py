import os
from pathlib import Path
import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb.errors import NotFoundError


if __name__=="__main__":


    env_loaded = dotenv.load_dotenv(Path.cwd().parent.parent.joinpath(".env"))
    if not env_loaded:
        raise FileNotFoundError("Environment file with OpenAI API key not found.")
    OA_TOKEN = os.getenv("OA_TOKEN")

    scenario_name = "arnhemdreijenseweg"
    # scenario_file = scenario_name+".md"
    # reader = SimpleDirectoryReader(input_dir="../data/scenarios", input_files=[scenario_file])
    # documents = reader.load_data()

    db = chromadb.PersistentClient(path=str(Path.cwd().parent.parent.joinpath("chroma_db")))
    # db.delete_collection(scenario_name)
    # chroma_collection = db.get_or_create_collection(scenario_name)
    try:
        chroma_collection = db.get_collection(scenario_name)
    except NotFoundError as err:
        err.add_note("\n Use `ingest_data` function to create a collection")
        raise err

    vector_store =  ChromaVectorStore(chroma_collection=chroma_collection)

    # pipeline = IngestionPipeline(
    #     transformations=[
    #         SentenceSplitter(chunk_size=8192, chunk_overlap=0),
    #         OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN),
    #     ],
    #     vector_store=vector_store
    # )
    #
    # nodes = pipeline.run(documents=documents)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=HF_TOKEN)
    llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=OA_TOKEN)

    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize", # other options are refine, compact
    )

    # Modify the player accordingly
    player = "German"
    file_name_to_save = f"{player}_direct_query_v0.0-2.txt"

    # response = query_engine.query("What information can you extract about Britain's military strength")
    # print(response)

    # system_prompt = """
    # You are an expert in assessing the strength of armed forces. From this scenario, you need to assess the military capability in numbers of the German armed forces.
    # Return the output as JSON response in the following format. Do not include any json headers in the response
    #
    # military_capability :
    # """

    system_prompt = f"""
        You are an expert in assessing the strength of armed forces. From this scenario, you need to assess the military capability in numbers of the {player} armed forces.
        """
    response = query_engine.query(system_prompt)
    print(response)


    with open(Path.cwd().parent.parent.joinpath(f"tmp/{file_name_to_save}"), "w+", encoding="utf-8") as file:
        file.write(f"System Prompt: {system_prompt}\n\n")
        file.write(f"Response: \n")
        file.write(response.response)
        file.close()


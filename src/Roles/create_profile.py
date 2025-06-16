
import os
from pathlib import Path
import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb.errors import NotFoundError

from src.Roles.utils.roles import Role


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
        streaming=False,
        llm=llm,
        response_mode="tree_summarize", # other options are refine, compact
    )

    # Initialise player
    player = "British"
    parameters = ["Military Capability"]
    role = Role(player=player, attributes=parameters)

    # This prompt is specific to military strength
    system_prompt = f"""
        You are an expert in assessing the strength of armed forces. From this scenario, you need to assess the military capability in numbers of the {player} armed forces.
        """
    response = query_engine.query(system_prompt)
    print(response)

    role.attributes[parameters[0]] = response.response

    role.save_to_file(directory=Path.cwd().parent.parent.joinpath("tmp"), scenario=scenario_name, file_suffix= "direct_query_v0.1")

    # TODO
    # System prompt templates specific to each value in the attributes list is required.

    # TODO
    # How to input the players and attributes (have to be the same for the participants? Not necessarily)
    # JSON or YAML file?




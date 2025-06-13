import os
import dotenv
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb



def ingest_data(scenario_name: str,
                path_to_scenario_file: Path,
                database: chromadb.PersistentClient,
                chunk_size: int,
                embedding_model: OpenAIEmbedding
                ):
    """
    This function stores the scenario into a Chromadb vector store

    Args:
        scenario_name: name using which a database collection is created
        path_to_scenario_file: path to the file of the scenario to store
        database: chroma database
        chunk_size: the size of the sentences into which the scenario will be split for storage and retrieval
        embedding_model: the OpenAI embedding model to be used to transform the text

    """

    try:
        doc_reader =  SimpleDirectoryReader(input_files=[path_to_scenario_file]
                                            )
        document = doc_reader.load_data()

        chroma_collection = database.get_or_create_collection(scenario_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size,
                                 chunk_overlap=0
                                 ),
                embedding_model
            ],
            vector_store=vector_store
        )

        pipeline.run(documents=document)

    except Exception as err:
        raise err


if __name__=="__main__":

    dotenv.load_dotenv(Path.cwd().parent.parent.parent.joinpath(".env"))
    OA_TOKEN = os.getenv("OA_TOKEN")

    scenario_name = "arnhemdreijenseweg"
    scenario_file = scenario_name + ".md"
    scenario_dir = Path.cwd().parent.parent.parent.joinpath("data/scenarios/md")

    embedding_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OA_TOKEN)

    chunk_size = 8192

    db = chromadb.PersistentClient(path=str(Path.cwd().parent.parent.parent.joinpath("chroma_db")))

    ingest_data(scenario_name=scenario_name,
                path_to_scenario_file=scenario_dir.joinpath(scenario_file),
                database=db,
                chunk_size=chunk_size,
                embedding_model=embedding_model
                )

import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()
nest_asyncio.apply()

CHROMA_DB_PATH = "./llm_chroma_db"
CHROMA_COLLECTION_NAME = "chroma"

def load_document():
    print("📂 Loading documents...")
    reader = SimpleDirectoryReader(input_dir="docs")
    documents = reader.load_data()
    print(f"✅ Loaded {len(documents)} documents")
    return documents

async def create_or_load_index():
    print("🔍 Checking if Chroma DB exists...")
    db_exists = os.path.exists(os.path.join(CHROMA_DB_PATH, "chroma.sqlite3"))

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = GeminiEmbedding(model_name="models/embedding-001")

    if not db_exists:
        print("🆕 Chroma DB not found. Creating and ingesting...")
        docs = load_document()

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),
                embed_model,
            ],
            vector_store=vector_store,
        )

        nodes = await pipeline.arun(documents=docs[:10])
        print(f"✅ Ingestion complete. {len(nodes)} chunks indexed.")
    else:
        print("✅ Chroma DB found. Skipping ingestion.")

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

async def create_query_engine():
    index = await create_or_load_index()
    llm = GoogleGenAI(model_name="gemini-2.0-flash", temperature=0.0)
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
        similarity_top_k=3,
        verbose=True
    )
    print("✅ Query engine is ready.")
    return query_engine

async def main():
    print("🚀 Starting...")
    query_engine = await create_query_engine()
    while True:
        query = input("\n🔍 Enter your query (type 'exit' to quit):\n")
        if query.lower() in {"exit", "end"}:
            print("👋 Exiting. Bye!")
            break
        print(f"Query: {query}")
        response = query_engine.query(query)
        print("\n💡 Response:\n", response.response)

if __name__ == "__main__":
    asyncio.run(main())

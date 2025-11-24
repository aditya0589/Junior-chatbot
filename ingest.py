import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

def ingest_data():
    # 1. Load Data
    loader = TextLoader("aditya.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Chunk Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    # Check if index exists, if not create it (optional, usually better to do manually to control specs)
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384, # Dimension for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # 5. Store in Pinecone
    print("Upserting vectors to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    # Check if index exists, and DELETE it if it does to handle dimension mismatch
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"Index {index_name} exists. Deleting to recreate with correct dimensions...")
        pc.delete_index(index_name)
        time.sleep(2) # Wait for deletion

    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=768, # Dimension for Google embedding-001
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
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    batch_size = 1
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        try:
            vectorstore.add_documents(batch)
            print("Batch added. Sleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Error adding batch: {e}. Sleeping for 30 seconds and retrying...")
            time.sleep(30)
            try:
                vectorstore.add_documents(batch)
                print("Retry successful.")
            except Exception as e2:
                print(f"Retry failed: {e2}. Skipping batch.")

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()

import os
import contextlib
import asyncio
try:
    import psutil
except ImportError:
    psutil = None

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Heavy imports moved to top-level to ensure they are loaded at startup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Global variable for the chain
rag_chain = None
init_task = None

async def initialize_rag_chain():
    global rag_chain
    print("STARTUP: Initializing RAG chain in background...")
    
    pid = os.getpid()
    if psutil:
        mem_before = psutil.Process(pid).memory_info().rss / 1024 ** 2
        print(f"Memory before init: {mem_before:.2f} MB")

    try:
        # 1. Initialize Embeddings (Google)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # 2. Initialize Vector Store
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # 3. Initialize Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. Initialize LLM (Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # 5. Create Chain
        system_prompt = (
            "You are Junior, a helpful assistant for Aditya. Your job is to introduce and answer questions about him to the users. Always introduce yourself as Junior, before giving any reply. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know and ask the user to contact Aditya. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        if psutil:
            mem_after = psutil.Process(pid).memory_info().rss / 1024 ** 2
            print(f"RAG Chain initialized. Memory after init: {mem_after:.2f} MB")
        else:
             print("RAG Chain initialized.")
             
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Start initialization in background
    global init_task
    init_task = asyncio.create_task(initialize_rag_chain())
    
    yield
    
    # Shutdown logic
    print("SHUTDOWN: Cleaning up...")

app = FastAPI(title="RAG Application", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Service is starting up, please try again in a few seconds.")
        
    try:
        response = rag_chain.invoke({"input": request.query})
        return {"answer": response["answer"], "context": [doc.page_content for doc in response["context"]]}
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    if rag_chain is None:
        return {"message": "RAG Application is starting up..."}
    return {"message": "RAG Application is running. Send POST requests to /query."}


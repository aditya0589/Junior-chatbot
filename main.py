import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="RAG Application")

class QueryRequest(BaseModel):
    query: str

# Global variables for the chain
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
    print("RAG Chain initialized.")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    
    response = rag_chain.invoke({"input": request.query})
    return {"answer": response["answer"], "context": [doc.page_content for doc in response["context"]]}

@app.get("/")
async def root():
    return {"message": "RAG Application is running. Send POST requests to /query."}

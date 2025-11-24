from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# Set dummy env vars before importing main to ensure no key errors if they are accessed at module level
os.environ["PINECONE_INDEX_NAME"] = "test-index"
os.environ["GOOGLE_API_KEY"] = "test-key"

# Patch the dependencies used in startup_event to avoid real connections
with patch("langchain_huggingface.HuggingFaceEmbeddings"), \
     patch("langchain_pinecone.PineconeVectorStore"), \
     patch("langchain_google_genai.ChatGoogleGenerativeAI"), \
     patch("langchain.chains.combine_documents.create_stuff_documents_chain"), \
     patch("langchain.chains.create_retrieval_chain"):
    from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "RAG Application is running. Send POST requests to /query."}

def test_query_endpoint_success():
    # Mock the rag_chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Test answer",
        "context": [MagicMock(page_content="Test context")]
    }
    
    # Patch the global rag_chain in main module
    with patch("main.rag_chain", mock_chain):
        response = client.post("/query", json={"query": "test question"})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Test answer",
            "context": ["Test context"]
        }

def test_query_endpoint_not_initialized():
    # Ensure rag_chain is None
    with patch("main.rag_chain", None):
        response = client.post("/query", json={"query": "test question"})
        assert response.status_code == 500
        assert response.json() == {"detail": "RAG chain not initialized"}

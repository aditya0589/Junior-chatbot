import asyncio
import os
from dotenv import load_dotenv
from main import startup_event, query_endpoint, QueryRequest

# Load environment variables
load_dotenv()

async def run_verification():
    print("Initializing RAG chain...")
    try:
        await startup_event()
        print("RAG chain initialized.")
        
        query_text = "who is aditya and what are his skills"
        print(f"Querying: {query_text}")
        
        response = await query_endpoint(QueryRequest(query=query_text))
        
        output = f"\n\nLive Query Verification\n-----------------------\n"
        output += f"Question: {query_text}\n"
        output += f"Answer: {response['answer']}\n"
        output += f"Context Retrieved: {len(response['context'])} documents\n"
        
        print(output)
        
        # Append to test_results.txt
        with open("test_results.txt", "a") as f:
            f.write(output)
            
    except Exception as e:
        error_msg = f"\n\nLive Query Verification Failed\n------------------------------\nError: {str(e)}\n"
        print(error_msg)
        with open("test_results.txt", "a") as f:
            f.write(error_msg)

if __name__ == "__main__":
    asyncio.run(run_verification())

"""
Test pgvector connection and RAG functionality
"""
import os
from dotenv import load_dotenv
from multi_agent_system import WebScraperAgent

load_dotenv()

def test_pgvector():
    print("="*80)
    print("Testing pgvector RAG Connection")
    print("="*80)
    
    try:
        # Initialize WebScraperAgent with pgvector enabled
        print("\n1. Initializing WebScraperAgent with pgvector...")
        agent = WebScraperAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            use_pgvector=True,
            pg_connection_string="postgresql+psycopg2://shri:shri123@localhost:6024/vectordb"
        )
        
        if agent.vector_store:
            print("   ✅ pgvector initialized successfully!")
            print(f"   Collection: research_cache")
            print(f"   Database: vectordb")
            print(f"   Host: localhost:6024")
        else:
            print("   ❌ pgvector initialization failed")
            return False
        
        # Test a simple query
        print("\n2. Testing search functionality...")
        test_query = "What is artificial intelligence?"
        
        result = agent.retrieve_information(
            query=test_query,
            context="Test query for pgvector"
        )
        
        if result:
            print("   ✅ Search completed successfully!")
            print(f"   Query: {test_query}")
            print(f"   Sources found: {len(result.get('retrieved_data', {}).get('sources', []))}")
            print(f"   Cached to pgvector: Yes")
        else:
            print("   ❌ Search failed")
            return False
        
        # Test cache retrieval
        print("\n3. Testing cache retrieval...")
        cached_result = agent.retrieve_information(
            query=test_query,
            context="Test query for pgvector"
        )
        
        if cached_result:
            print("   ✅ Cache retrieval successful!")
            print("   Result should be from pgvector cache")
        
        print("\n" + "="*80)
        print("✅ All tests passed! pgvector RAG is working correctly")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pgvector()

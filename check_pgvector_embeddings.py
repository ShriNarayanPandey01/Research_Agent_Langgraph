"""
Script to inspect embeddings stored in pgvector database
Shows various ways to query and analyze the vector data
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 6024,
    'database': 'vectordb',
    'user': 'shri',
    'password': 'shri123'
}

def connect_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def check_extensions(conn):
    """Check if vector extension is enabled"""
    print("="*80)
    print("1. CHECKING EXTENSIONS")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        
        if result:
            print(f"✅ Vector extension is installed")
            print(f"   Version: {result['extversion']}")
        else:
            print("❌ Vector extension not found!")
            print("   Run: CREATE EXTENSION vector;")
    print()

def list_collections(conn):
    """List all tables (collections) in the database"""
    print("="*80)
    print("2. LISTING ALL TABLES/COLLECTIONS")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        
        if tables:
            print(f"Found {len(tables)} table(s):")
            for i, table in enumerate(tables, 1):
                print(f"   {i}. {table['table_name']}")
        else:
            print("No tables found")
    print()

def inspect_collection_schema(conn, collection_name='langchain_pg_collection'):
    """Inspect the schema of a collection"""
    print("="*80)
    print(f"3. INSPECTING SCHEMA: {collection_name}")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{collection_name}'
            );
        """)
        
        exists = cur.fetchone()['exists']
        
        if not exists:
            print(f"❌ Table '{collection_name}' does not exist")
            return
        
        # Get column information
        cur.execute(f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{collection_name}'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        
        print(f"✅ Table structure:")
        for col in columns:
            length = f"({col['character_maximum_length']})" if col['character_maximum_length'] else ""
            print(f"   - {col['column_name']}: {col['data_type']}{length}")
    print()

def count_embeddings(conn, collection_name='langchain_pg_embedding'):
    """Count total embeddings stored"""
    print("="*80)
    print(f"4. COUNTING EMBEDDINGS IN: {collection_name}")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{collection_name}'
            );
        """)
        
        exists = cur.fetchone()['exists']
        
        if not exists:
            print(f"❌ Table '{collection_name}' does not exist yet")
            print("   (Embeddings will be created on first use)")
            return 0
        
        # Count rows
        cur.execute(f"SELECT COUNT(*) as count FROM {collection_name};")
        count = cur.fetchone()['count']
        
        print(f"✅ Total embeddings stored: {count}")
        
        if count > 0:
            # Get embedding dimension
            cur.execute(f"""
                SELECT embedding 
                FROM {collection_name} 
                LIMIT 1;
            """)
            result = cur.fetchone()
            if result and result['embedding']:
                dimension = len(result['embedding'])
                print(f"   Embedding dimension: {dimension}")
    print()
    return count

def view_recent_embeddings(conn, collection_name='langchain_pg_embedding', limit=5):
    """View recent embeddings with metadata"""
    print("="*80)
    print(f"5. VIEWING RECENT EMBEDDINGS (Last {limit})")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{collection_name}'
            );
        """)
        
        exists = cur.fetchone()['exists']
        
        if not exists:
            print(f"❌ Table '{collection_name}' does not exist yet")
            return
        
        # Get recent embeddings
        cur.execute(f"""
            SELECT 
                id,
                document,
                cmetadata,
                LENGTH(embedding::text) as embedding_size
            FROM {collection_name}
            ORDER BY id DESC
            LIMIT {limit};
        """)
        
        embeddings = cur.fetchall()
        
        if not embeddings:
            print("No embeddings found")
            return
        
        for i, emb in enumerate(embeddings, 1):
            print(f"\n📄 Embedding {i}:")
            print(f"   ID: {emb['id']}")
            print(f"   Document (first 100 chars): {emb['document'][:100]}...")
            
            if emb['cmetadata']:
                print(f"   Metadata:")
                metadata = emb['cmetadata']
                for key, value in metadata.items():
                    if key == 'query':
                        print(f"      - {key}: {value}")
                    elif key == 'timestamp':
                        print(f"      - {key}: {value}")
                    elif key == 'query_hash':
                        print(f"      - {key}: {value[:16]}...")
            
            print(f"   Embedding size: {emb['embedding_size']} chars")
    print()

def search_by_metadata(conn, collection_name='langchain_pg_embedding', query_hash=None):
    """Search embeddings by metadata"""
    print("="*80)
    print("6. SEARCHING BY METADATA")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{collection_name}'
            );
        """)
        
        exists = cur.fetchone()['exists']
        
        if not exists:
            print(f"❌ Table '{collection_name}' does not exist yet")
            return
        
        if query_hash:
            # Search by specific query_hash
            cur.execute(f"""
                SELECT id, cmetadata->>'query' as query, cmetadata->>'timestamp' as timestamp
                FROM {collection_name}
                WHERE cmetadata->>'query_hash' = %s;
            """, (query_hash,))
        else:
            # Show all unique queries
            cur.execute(f"""
                SELECT DISTINCT 
                    cmetadata->>'query' as query,
                    cmetadata->>'query_hash' as query_hash,
                    COUNT(*) as count
                FROM {collection_name}
                GROUP BY cmetadata->>'query', cmetadata->>'query_hash'
                ORDER BY count DESC;
            """)
        
        results = cur.fetchall()
        
        if results:
            print(f"Found {len(results)} unique queries:")
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. Query: {result['query']}")
                if 'query_hash' in result:
                    print(f"      Hash: {result['query_hash'][:16]}...")
                if 'count' in result:
                    print(f"      Cached: {result['count']} time(s)")
        else:
            print("No results found")
    print()

def get_embedding_stats(conn, collection_name='langchain_pg_embedding'):
    """Get statistics about embeddings"""
    print("="*80)
    print("7. EMBEDDING STATISTICS")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{collection_name}'
            );
        """)
        
        exists = cur.fetchone()['exists']
        
        if not exists:
            print(f"❌ Table '{collection_name}' does not exist yet")
            return
        
        # Total count
        cur.execute(f"SELECT COUNT(*) as total FROM {collection_name};")
        total = cur.fetchone()['total']
        
        if total == 0:
            print("No embeddings stored yet")
            return
        
        print(f"📊 Statistics:")
        print(f"   Total embeddings: {total}")
        
        # Unique queries
        cur.execute(f"""
            SELECT COUNT(DISTINCT cmetadata->>'query') as unique_queries
            FROM {collection_name}
            WHERE cmetadata->>'query' IS NOT NULL;
        """)
        unique = cur.fetchone()['unique_queries']
        print(f"   Unique queries: {unique}")
        
        # Date range
        cur.execute(f"""
            SELECT 
                MIN(cmetadata->>'timestamp') as earliest,
                MAX(cmetadata->>'timestamp') as latest
            FROM {collection_name}
            WHERE cmetadata->>'timestamp' IS NOT NULL;
        """)
        dates = cur.fetchone()
        if dates['earliest']:
            print(f"   Earliest: {dates['earliest']}")
            print(f"   Latest: {dates['latest']}")
    print()

def main():
    print("\n" + "="*80)
    print("🔍 PGVECTOR EMBEDDINGS INSPECTOR")
    print("="*80 + "\n")
    
    conn = connect_db()
    if not conn:
        return
    
    try:
        # Run all inspections
        check_extensions(conn)
        list_collections(conn)
        inspect_collection_schema(conn)
        
        count = count_embeddings(conn)
        
        if count > 0:
            view_recent_embeddings(conn, limit=3)
            search_by_metadata(conn)
            get_embedding_stats(conn)
        
        print("="*80)
        print("✅ INSPECTION COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()

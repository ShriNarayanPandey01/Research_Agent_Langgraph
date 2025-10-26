# pgvector RAG Implementation

## Overview
Successfully integrated **pgvector** (PostgreSQL vector database) for RAG (Retrieval-Augmented Generation) functionality, replacing the previous file-based caching system with a production-ready vector database.

## Docker Container Configuration
```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=shri \
  -e POSTGRES_PASSWORD=shri123 \
  -e POSTGRES_DB=vectordb \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
```

**Connection Details:**
- Host: `localhost`
- Port: `6024`
- Database: `vectordb`
- User: `shri`
- Password: `shri123`
- Connection String: `postgresql+psycopg2://shri:shri123@localhost:6024/vectordb`

## Features Implemented

### 1. **Vector Database RAG Cache**
- **Primary Storage**: pgvector for semantic search and caching
- **Fallback Storage**: File-based cache (`.web_scraper_cache/`) for redundancy
- **Memory Cache**: In-memory cache for fastest retrieval

### 2. **Three-Tier Caching System**
```
1. Memory Cache (fastest) → 2. pgvector (semantic search) → 3. Disk Cache (fallback)
```

### 3. **Semantic Search Capability**
- Uses OpenAI embeddings for vector representations
- Stores research results with metadata (query, timestamp, hash)
- Enables similarity-based retrieval for related queries

## Code Changes

### Updated Files:
1. **`requirements.txt`**
   - Added: `pgvector>=0.2.0`
   - Added: `psycopg2-binary>=2.9.0`
   - Added: `langchain-postgres>=0.0.6`
   - Commented out: ChromaDB dependencies (kept as backup)

2. **`multi_agent_system.py`**
   - Replaced `from langchain_chroma import Chroma` with `from langchain_postgres import PGVector`
   - Updated `WebScraperAgent.__init__()` to initialize pgvector
   - Modified `_get_from_cache()` to search pgvector with semantic similarity
   - Modified `_save_to_cache()` to store in both pgvector and disk
   - Added connection string parameter for flexibility

### New Features in WebScraperAgent:
```python
WebScraperAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    use_pgvector=True,  # Enable/disable pgvector
    pg_connection_string="postgresql+psycopg2://shri:shri123@localhost:6024/vectordb"
)
```

## Testing

### Test Script: `test_pgvector_connection.py`
Validates:
- ✅ pgvector connection initialization
- ✅ Vector store creation with collection name
- ✅ Document storage with embeddings
- ✅ Semantic search retrieval
- ✅ Cache hit/miss detection
- ✅ Memory, pgvector, and disk cache fallback

### Test Results:
```
✅ pgvector initialized successfully!
✅ Search completed successfully! (3 sources found)
✅ Cache retrieval successful! (from pgvector)
```

## Benefits

### 1. **Production-Ready**
- PostgreSQL is battle-tested for production workloads
- pgvector extension is actively maintained
- ACID compliance for data integrity

### 2. **Scalability**
- Can handle millions of vectors
- Horizontal scaling with PostgreSQL replication
- Efficient indexing (HNSW, IVFFlat)

### 3. **Semantic Search**
- Find similar queries even with different wording
- Leverage OpenAI embeddings for meaning-based retrieval
- Better cache hit rates through similarity matching

### 4. **Redundancy**
- Three-tier caching ensures no data loss
- Graceful fallback if pgvector is unavailable
- File-based backup for offline access

## Usage

### Running the System:
```python
from multi_agent_system import ManagerAgent

# Initialize with pgvector enabled (default)
manager = ManagerAgent()

# Run research
results = manager.orchestrate_research(
    query="Your research question",
    page_limit=3
)
```

### Testing pgvector:
```bash
python test_pgvector_connection.py
```

### Running Demo:
```bash
python demo_text_report_terminal.py
```

## Architecture

### Vector Store Schema:
- **Collection**: `research_cache`
- **Embeddings**: OpenAI `text-embedding-ada-002`
- **Metadata Fields**:
  - `query`: Original search query
  - `query_hash`: MD5 hash for exact matching
  - `timestamp`: ISO 8601 timestamp
  - `type`: Document type (`research_cache`)

### Data Flow:
```
Query → Hash Generation → Memory Check → pgvector Semantic Search → 
Disk Fallback → Fresh Web Search → Store in All Three Caches
```

## Dependencies Installed
```
pgvector==0.3.6
psycopg2-binary==2.9.11
langchain-postgres==0.0.16
asyncpg==0.30.0
psycopg==3.2.11
psycopg-pool==3.2.6
```

## Future Enhancements
- [ ] Add index optimization (HNSW for faster similarity search)
- [ ] Implement cache expiration policies
- [ ] Add analytics dashboard for cache performance
- [ ] Support multiple embedding models
- [ ] Implement distributed pgvector cluster

## Troubleshooting

### Connection Issues:
If pgvector fails to connect, the system automatically falls back to file-based cache.

### Check Docker Container:
```bash
docker ps | grep pgvector
docker logs pgvector-container
```

### Test Connection Manually:
```bash
psql -h localhost -p 6024 -U shri -d vectordb
\dx  # List extensions, should show 'vector'
```

## Summary
✅ pgvector successfully integrated with PostgreSQL  
✅ Three-tier caching (memory → pgvector → disk)  
✅ Semantic search with OpenAI embeddings  
✅ Production-ready with graceful fallbacks  
✅ All tests passing  

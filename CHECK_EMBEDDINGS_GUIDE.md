# How to Check pgvector Embeddings

## Method 1: Using Python Script (Recommended)
```bash
python check_pgvector_embeddings.py
```

**What it shows:**
- ✅ Extension status
- ✅ Table structure
- ✅ Total embeddings count (33 embeddings found!)
- ✅ Recent embeddings with metadata
- ✅ All unique queries cached
- ✅ Statistics (earliest/latest timestamps)

---

## Method 2: Using Docker + psql (Command Line)

### Connect to PostgreSQL inside Docker:
```bash
docker exec -it pgvector-container psql -U shri -d vectordb
```

### Once connected, run these SQL queries:

#### 1. Check vector extension:
```sql
\dx
```
**Output:** Should show `vector | 0.8.1`

#### 2. List all tables:
```sql
\dt
```
**Output:** Shows `langchain_pg_collection` and `langchain_pg_embedding`

#### 3. Count total embeddings:
```sql
SELECT COUNT(*) FROM langchain_pg_embedding;
```
**Output:** `33` (as of now)

#### 4. View table structure:
```sql
\d langchain_pg_embedding
```
**Output:** Shows columns: id, embedding, document, cmetadata, etc.

#### 5. View recent embeddings:
```sql
SELECT 
    id,
    LEFT(document, 80) as doc_preview,
    cmetadata->>'query' as query,
    cmetadata->>'timestamp' as cached_at
FROM langchain_pg_embedding
ORDER BY id DESC
LIMIT 5;
```

#### 6. Count unique queries:
```sql
SELECT COUNT(DISTINCT cmetadata->>'query') FROM langchain_pg_embedding;
```

#### 7. View all cached queries:
```sql
SELECT 
    cmetadata->>'query' as query,
    cmetadata->>'timestamp' as cached_at,
    LEFT(cmetadata->>'query_hash', 16) as hash_preview
FROM langchain_pg_embedding
ORDER BY cmetadata->>'timestamp' DESC;
```

#### 8. Get embedding dimension:
```sql
SELECT array_length(embedding, 1) as dimension 
FROM langchain_pg_embedding 
LIMIT 1;
```
**Output:** `19438` (OpenAI embedding dimension)

#### 9. Check specific query:
```sql
SELECT 
    cmetadata->>'query' as query,
    cmetadata->>'timestamp' as timestamp,
    LENGTH(document) as doc_size
FROM langchain_pg_embedding
WHERE cmetadata->>'query' LIKE '%artificial intelligence%';
```

#### 10. View metadata statistics:
```sql
SELECT 
    COUNT(*) as total_embeddings,
    COUNT(DISTINCT cmetadata->>'query') as unique_queries,
    MIN(cmetadata->>'timestamp') as first_cached,
    MAX(cmetadata->>'timestamp') as last_cached
FROM langchain_pg_embedding;
```

#### 11. Exit psql:
```sql
\q
```

---

## Method 3: Using DBeaver/pgAdmin (GUI Tools)

### DBeaver:
1. Download: https://dbeaver.io/download/
2. Connect to:
   - Host: `localhost`
   - Port: `6024`
   - Database: `vectordb`
   - User: `shri`
   - Password: `shri123`
3. Navigate to: `vectordb → Schemas → public → Tables`
4. Right-click table → View Data

### pgAdmin:
1. Download: https://www.pgadmin.org/download/
2. Add New Server:
   - Name: `pgvector-local`
   - Host: `localhost`
   - Port: `6024`
   - Database: `vectordb`
   - Username: `shri`
   - Password: `shri123`
3. Browse: Servers → pgvector-local → Databases → vectordb → Schemas → public → Tables

---

## Method 4: Quick Docker Command (One-liner)

### Count embeddings:
```bash
docker exec -it pgvector-container psql -U shri -d vectordb -c "SELECT COUNT(*) FROM langchain_pg_embedding;"
```

### View recent 5 embeddings:
```bash
docker exec -it pgvector-container psql -U shri -d vectordb -c "SELECT cmetadata->>'query' as query FROM langchain_pg_embedding ORDER BY id DESC LIMIT 5;"
```

### Check vector extension:
```bash
docker exec -it pgvector-container psql -U shri -d vectordb -c "\dx"
```

---

## Current Status (Your Database)

Based on the inspection:

✅ **Total Embeddings:** 33  
✅ **Unique Queries:** 33  
✅ **Vector Dimension:** 19,438 (OpenAI ada-002)  
✅ **Extension:** vector 0.8.1  
✅ **Tables:** langchain_pg_collection, langchain_pg_embedding  
✅ **Date Range:** Oct 25, 2025 (19:00 - 19:42)  

### Sample Queries Cached:
- "What is artificial intelligence?"
- "Analyze the overall economic impact of COVID-19 on developing countries"
- "Examine the impact of COVID-19 on employment rates"
- Multiple fact-checking queries
- And 29 more research queries...

---

## Useful Queries for Analysis

### Find embeddings by keyword:
```sql
SELECT cmetadata->>'query' 
FROM langchain_pg_embedding 
WHERE cmetadata->>'query' ILIKE '%COVID%';
```

### Get cache hit rate (if tracking):
```sql
SELECT 
    cmetadata->>'cached' as is_cached,
    COUNT(*) 
FROM langchain_pg_embedding 
GROUP BY cmetadata->>'cached';
```

### View embedding storage size:
```sql
SELECT 
    pg_size_pretty(pg_total_relation_size('langchain_pg_embedding')) as total_size;
```

---

## Tips

1. **Performance**: Use indexes on metadata for faster queries
2. **Cleanup**: Delete old embeddings if needed:
   ```sql
   DELETE FROM langchain_pg_embedding 
   WHERE (cmetadata->>'timestamp')::timestamp < NOW() - INTERVAL '30 days';
   ```
3. **Backup**: Export embeddings:
   ```bash
   docker exec pgvector-container pg_dump -U shri vectordb > backup.sql
   ```

---

## Troubleshooting

### Can't connect?
```bash
docker ps | grep pgvector  # Check if container is running
docker logs pgvector-container  # Check logs
```

### Extension not found?
```sql
CREATE EXTENSION vector;
```

### Table doesn't exist?
- Tables are created automatically on first use
- Run `python test_pgvector_connection.py` to create them

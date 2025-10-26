"""
Multi-Agent Research System
This implements a sophisticated multi-agent system with:
1. Manager Agent (Research Orchestrator)
2. Web Scraper & Document Retrieval Agent (with Vector DB RAG)
3. Deep Analysis Agent
4. Fact Checking & Validation Agent
5. Output Formatting Agent
"""

from typing import TypedDict, List, Dict, Any, Annotated, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore
from pydantic import BaseModel, Field
import json
from datetime import datetime
import operator
import hashlib
import pickle
import os
from pathlib import Path
from openai import OpenAI
import uuid


# ============================================================================
# DATA MODELS
# ============================================================================

class SubTask(BaseModel):
    """Model for a research sub-task"""
    id: str
    query: str
    status: str = "pending"  # pending, analyzing, fact_checking, completed
    priority: int
    deep_analysis_result: Dict[str, Any] = Field(default_factory=dict)
    fact_check_result: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Model for agent responses"""
    agent_name: str
    task_id: str
    success: bool
    result: Dict[str, Any]
    timestamp: str
    confidence: float = 0.0


# ============================================================================
# AGENT 1: WEB SCRAPER & DOCUMENT RETRIEVAL AGENT (WITH RAG)
# ============================================================================

class WebScraperAgent:
    """
    Agent responsible for fetching information from web sources and documents
    Features:
    - RAG (Retrieval-Augmented Generation) for caching search results
    - OpenAI Web Search integration for real-time data
    - Persistent storage to avoid redundant searches
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = ".web_scraper_cache", 
                 use_pgvector: bool = True, pg_connection_string: str = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
        self.openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.name = "WebScraperAgent"
        
        # Initialize embeddings for RAG
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Initialize pgvector RAG if enabled
        self.use_pgvector = use_pgvector
        self.vector_store = None
        
        if use_pgvector:
            # Default connection string for your pgvector container
            if pg_connection_string is None:
                pg_connection_string = (
                    "postgresql+psycopg2://shri:shri123@localhost:6024/vectordb"
                )
            
            try:
                self.vector_store = PGVectorStore(
                    embeddings=self.embeddings,
                    collection_name="research_cache",
                    connection=pg_connection_string,
                    use_jsonb=True,
                )
                print(f"   📚 pgvector RAG initialized: Connected to PostgreSQL")
            except Exception as e:
                print(f"   ⚠️  pgvector initialization failed: {e}")
                print(f"   📚 Falling back to file-based cache")
                self.use_pgvector = False
        
        # Fallback: File-based cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for quick lookups
        self._memory_cache = {}
        
        # Load existing cache index (for file-based cache)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        if not use_pgvector or self.vector_store is None:
            print(f"   📚 File Cache initialized: {len(self.cache_index)} entries loaded")
    
    def _load_cache_index(self) -> Dict[str, Dict]:
        """Load the cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save the cache index to disk"""
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_query_hash(self, query: str, context: str = "") -> str:
        """Generate a hash for the query to use as cache key"""
        combined = f"{query}|{context}".lower().strip()
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_from_cache(self, query_hash: str, query: str = "") -> Dict[str, Any]:
        """Retrieve cached results if available"""
        # Check memory cache first
        if query_hash in self._memory_cache:
            print(f"     💾 Cache HIT (memory): {query_hash[:8]}...")
            return self._memory_cache[query_hash]
        
        # Check pgvector if enabled
        if self.use_pgvector and self.vector_store:
            try:
                # Search for similar queries in vector store
                docs = self.vector_store.similarity_search(
                    query if query else query_hash, 
                    k=1,
                    filter={"query_hash": query_hash}
                )
                
                if docs and len(docs) > 0:
                    cached_data = json.loads(docs[0].page_content)
                    self._memory_cache[query_hash] = cached_data
                    print(f"     💾 Cache HIT (pgvector): {query_hash[:8]}...")
                    return cached_data
            except Exception as e:
                print(f"     ⚠️  pgvector search error: {e}")
        
        # Fallback to disk cache
        if query_hash in self.cache_index:
            cache_file = self.cache_dir / f"{query_hash}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Load into memory cache
                    self._memory_cache[query_hash] = cached_data
                    
                    print(f"     💾 Cache HIT (disk): {query_hash[:8]}...")
                    print(f"        Cached on: {self.cache_index[query_hash]['timestamp']}")
                    
                    return cached_data
                except:
                    pass
        
        print(f"     🔍 Cache MISS: Fetching fresh data...")
        return None
    
    def _save_to_cache(self, query_hash: str, query: str, data: Dict[str, Any]):
        """Save results to cache"""
        # Save to memory
        self._memory_cache[query_hash] = data
        
        # Save to pgvector if enabled
        if self.use_pgvector and self.vector_store:
            try:
                doc = Document(
                    page_content=json.dumps(data, ensure_ascii=False),
                    metadata={
                        "query": query,
                        "query_hash": query_hash,
                        "timestamp": datetime.now().isoformat(),
                        "type": "research_cache"
                    }
                )
                self.vector_store.add_documents([doc])
                print(f"     💾 Cached to pgvector: {query_hash[:8]}...")
            except Exception as e:
                print(f"     ⚠️  pgvector save error: {e}, falling back to disk")
        
        # Also save to disk as backup
        cache_file = self.cache_dir / f"{query_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Update index
        self.cache_index[query_hash] = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "file": f"{query_hash}.pkl"
        }
        self._save_cache_index()
        
        if not (self.use_pgvector and self.vector_store):
            print(f"     💾 Cached to disk: {query_hash[:8]}...")
    
    def _search_web_openai(self, query: str) -> Dict[str, Any]:
        """
        Use OpenAI's web search tool to fetch real-time information
        """
        try:
            print(f"     🌐 Performing OpenAI web search...")
            
            # Use OpenAI's web search capability
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a web search agent. Search the web and provide comprehensive information.
                        
                        Return your response as JSON with this structure:
                        {
                            "sources": [
                                {
                                    "source_name": "Source name",
                                    "url": "URL",
                                    "credibility": "high/medium/low",
                                    "content": "Retrieved content",
                                    "date": "Publication date"
                                }
                            ],
                            "summary": "Brief summary of findings",
                            "key_facts": ["fact1", "fact2"],
                            "data_points": {}
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Search the web for: {query}\n\nProvide comprehensive, up-to-date information."
                    }
                ]
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                print(f"     ✅ Web search complete: {len(result.get('sources', []))} sources found")
                return result
            except:
                # Fallback structure
                return {
                    "sources": [{"source_name": "OpenAI Search", "content": content}],
                    "summary": content[:200],
                    "key_facts": [],
                    "data_points": {}
                }
        
        except Exception as e:
            print(f"     ⚠️  Web search failed: {e}")
            print(f"     ↪️  Falling back to LLM knowledge...")
            
            # Fallback to LLM's built-in knowledge
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """Fallback to LLM's built-in knowledge if web search fails"""
        system_prompt = """You are a knowledge retrieval agent.
        
        Provide comprehensive information from your knowledge base. Include:
        1. Key facts and data points
        2. Multiple perspectives
        3. Context and background
        
        Return as JSON:
        {
            "sources": [{"source_name": "...", "content": "...", "credibility": "..."}],
            "summary": "...",
            "key_facts": [...],
            "data_points": {}
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except:
            return {
                "sources": [{"source_name": "Knowledge Base", "content": response.content}],
                "summary": response.content[:200],
                "key_facts": [],
                "data_points": {}
            }
    
    def retrieve_information(self, query: str, context: str = "", force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves relevant information for a query with RAG caching
        
        Args:
            query: The search query
            context: Additional context for the query
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary with retrieved information
        """
        # Generate cache key
        query_hash = self._get_query_hash(query, context)
        
        # Check cache (unless force refresh)
        if not force_refresh:
            cached_result = self._get_from_cache(query_hash, query)
            if cached_result:
                return cached_result
        
        # Fetch fresh data using OpenAI web search
        retrieved_data = self._search_web_openai(query)
        
        # Prepare result
        result = {
            "agent": self.name,
            "query": query,
            "retrieved_data": retrieved_data,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
        
        # Save to cache
        self._save_to_cache(query_hash, query, result)
        
        return result
    
    def clear_cache(self):
        """Clear all cached data"""
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear disk cache
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        
        self.cache_index = {}
        self._save_cache_index()
        
        print(f"   🗑️  Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        return {
            "total_entries": len(self.cache_index),
            "memory_entries": len(self._memory_cache),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


# ============================================================================
# AGENT 2: DEEP ANALYSIS AGENT (WITH ADVANCED ANALYSIS TOOLS)
# ============================================================================

class DeepAnalysisAgent:
    """
    Agent responsible for performing deep analysis on retrieved information
    
    Enhanced with 4 specialized analysis tools:
    1. Comparative Analysis - Compare entities, concepts, approaches
    2. Trend Analysis - Identify patterns and trends over time
    3. Causal Reasoning - Determine cause-effect relationships
    4. Statistical Analysis - Analyze data patterns and distributions
    
    Works with WebScraperAgent to get data and analyze it
    """
    
    def __init__(self, api_key: str = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
        self.name = "DeepAnalysisAgent"
        self.web_scraper = WebScraperAgent(api_key=api_key)
        
        # Initialize analysis tools
        print(f"   🔧 Initializing {self.name} with 4 advanced tools...")
        self._setup_analysis_tools()
    
    def _setup_analysis_tools(self):
        """Setup the four specialized analysis tools"""
        self.tools = {
            "comparative_analysis": self._comparative_analysis_tool,
            "trend_analysis": self._trend_analysis_tool,
            "causal_reasoning": self._causal_reasoning_tool,
            "statistical_analysis": self._statistical_analysis_tool
        }
        print(f"      ✅ Comparative Analysis Tool loaded")
        print(f"      ✅ Trend Analysis Tool loaded")
        print(f"      ✅ Causal Reasoning Tool loaded")
        print(f"      ✅ Statistical Analysis Tool loaded")
    
    def _comparative_analysis_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 1: Comparative Analysis
        Compares multiple entities, concepts, or approaches
        
        Returns:
            - Similarities and differences
            - Strengths and weaknesses comparison
            - Relative rankings or scores
            - Recommendation based on comparison
        """
        print(f"        🔄 Running Comparative Analysis...")
        
        system_prompt = """You are a Comparative Analysis expert. Analyze and compare the information provided.

        Perform systematic comparison across:
        1. Feature comparison - What each entity offers
        2. Pros/cons analysis - Advantages and disadvantages
        3. Performance comparison - Efficiency, effectiveness
        4. Use case suitability - Best fit scenarios
        
        Return JSON:
        {
            "comparison_matrix": {
                "entities": ["entity1", "entity2", ...],
                "dimensions": ["feature1", "feature2", ...],
                "scores": [[score11, score12, ...], ...]
            },
            "similarities": ["similarity1", ...],
            "differences": ["difference1", ...],
            "strengths_weaknesses": {
                "entity1": {"strengths": [...], "weaknesses": [...]},
                "entity2": {"strengths": [...], "weaknesses": [...]}
            },
            "ranking": ["1st: entity", "2nd: entity", ...],
            "recommendation": "Based on analysis...",
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze and compare:\n{json.dumps(data, indent=2)}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "comparison_matrix": {},
                "similarities": [],
                "differences": [],
                "recommendation": response.content[:500],
                "confidence": 0.6
            }
    
    def _trend_analysis_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 2: Trend Analysis
        Identifies patterns, trends, and temporal changes
        
        Returns:
            - Emerging trends
            - Growth/decline patterns
            - Momentum indicators
            - Future projections
        """
        print(f"        📈 Running Trend Analysis...")
        
        system_prompt = """You are a Trend Analysis expert. Identify patterns and trends in the data.

        Analyze:
        1. Historical trends - What has been happening
        2. Current state - Where things stand now
        3. Emerging patterns - New developments
        4. Future projections - What's likely to happen
        5. Momentum - Speed and direction of change
        
        Return JSON:
        {
            "historical_trends": ["trend1", "trend2", ...],
            "current_state": "Description of current situation",
            "emerging_patterns": [
                {
                    "pattern": "pattern description",
                    "strength": "weak/moderate/strong",
                    "impact": "low/medium/high"
                }
            ],
            "trend_direction": "upward/downward/stable/volatile",
            "momentum": {
                "speed": "slow/moderate/fast",
                "acceleration": "increasing/stable/decreasing"
            },
            "future_projection": {
                "short_term": "3-6 months outlook",
                "medium_term": "6-12 months outlook",
                "long_term": "1-3 years outlook"
            },
            "key_drivers": ["driver1", "driver2", ...],
            "risks": ["risk1", "risk2", ...],
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Identify trends in:\n{json.dumps(data, indent=2)}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "historical_trends": [],
                "emerging_patterns": [],
                "trend_direction": "stable",
                "future_projection": {"short_term": response.content[:300]},
                "confidence": 0.6
            }
    
    def _causal_reasoning_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 3: Causal Reasoning
        Determines cause-effect relationships and dependencies
        
        Returns:
            - Causal chains
            - Root causes
            - Contributing factors
            - Downstream effects
        """
        print(f"        🧩 Running Causal Reasoning...")
        
        system_prompt = """You are a Causal Reasoning expert. Identify cause-effect relationships.

        Analyze:
        1. Root causes - Fundamental underlying factors
        2. Direct causes - Immediate triggers
        3. Contributing factors - Supporting elements
        4. Causal chains - X causes Y causes Z
        5. Downstream effects - Consequences and impacts
        6. Mediating factors - What influences the relationship
        
        Return JSON:
        {
            "causal_chains": [
                {
                    "cause": "root cause",
                    "mechanisms": ["mechanism1", ...],
                    "effects": ["effect1", "effect2", ...],
                    "strength": "weak/moderate/strong",
                    "evidence": "supporting evidence"
                }
            ],
            "root_causes": [
                {
                    "cause": "fundamental cause",
                    "importance": "high/medium/low",
                    "explanation": "why this is a root cause"
                }
            ],
            "contributing_factors": ["factor1", "factor2", ...],
            "mediating_variables": ["variable1", ...],
            "confounding_factors": ["factor1", ...],
            "causal_graph": {
                "nodes": ["factor1", "factor2", ...],
                "edges": [{"from": "A", "to": "B", "strength": 0.8}, ...]
            },
            "alternative_explanations": ["explanation1", ...],
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Determine causal relationships in:\n{json.dumps(data, indent=2)}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "causal_chains": [],
                "root_causes": [],
                "contributing_factors": [],
                "alternative_explanations": [response.content[:300]],
                "confidence": 0.6
            }
    
    def _statistical_analysis_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 4: Statistical Analysis
        Analyzes data patterns, distributions, and statistical properties
        
        Returns:
            - Descriptive statistics
            - Distribution patterns
            - Correlations
            - Outliers and anomalies
        """
        print(f"        📊 Running Statistical Analysis...")
        
        system_prompt = """You are a Statistical Analysis expert. Analyze data patterns and statistics.

        Analyze:
        1. Descriptive statistics - Mean, median, mode, range
        2. Distribution patterns - Normal, skewed, bimodal
        3. Correlations - Relationships between variables
        4. Outliers - Unusual or anomalous data points
        5. Variability - Spread and consistency
        6. Statistical significance - Meaningful differences
        
        Return JSON:
        {
            "descriptive_stats": {
                "sample_size": 100,
                "central_tendency": {
                    "mean": "value",
                    "median": "value",
                    "mode": "value"
                },
                "dispersion": {
                    "range": "min-max",
                    "std_dev": "value",
                    "variance": "value"
                }
            },
            "distribution": {
                "type": "normal/skewed/bimodal/uniform",
                "skewness": "left/right/symmetric",
                "kurtosis": "leptokurtic/mesokurtic/platykurtic"
            },
            "patterns": [
                {
                    "pattern": "pattern description",
                    "frequency": "high/medium/low",
                    "significance": "very significant/significant/minor"
                }
            ],
            "correlations": [
                {
                    "variables": ["var1", "var2"],
                    "strength": 0.0-1.0,
                    "direction": "positive/negative",
                    "significance": "high/medium/low"
                }
            ],
            "outliers": [
                {
                    "value": "outlier description",
                    "deviation": "how far from norm",
                    "potential_cause": "explanation"
                }
            ],
            "statistical_tests": {
                "test_performed": "test name",
                "p_value": "value",
                "conclusion": "accept/reject hypothesis"
            },
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Perform statistical analysis on:\n{json.dumps(data, indent=2)}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "descriptive_stats": {},
                "distribution": {"type": "unknown"},
                "patterns": [],
                "correlations": [],
                "outliers": [],
                "confidence": 0.6
            }
    
    def analyze_task(self, task: SubTask, use_tools: List[str] = None) -> Dict[str, Any]:
        """
        Performs deep analysis on a task by:
        1. Using WebScraperAgent to retrieve information
        2. Running applicable analysis tools
        3. Synthesizing results into comprehensive analysis
        
        Args:
            task: SubTask to analyze
            use_tools: List of tool names to use (default: all tools)
                      Options: ["comparative_analysis", "trend_analysis", 
                               "causal_reasoning", "statistical_analysis"]
        """
        print(f"  🔬 {self.name}: Starting analysis for task {task.id}")
        
        # Step 1: Retrieve information using WebScraperAgent
        print(f"     📡 Fetching information via WebScraperAgent...")
        retrieved_data = self.web_scraper.retrieve_information(
            query=task.query,
            context=f"Priority: {task.priority}"
        )
        
        # Step 2: Determine which tools to use
        if use_tools is None:
            # Automatically select tools based on query keywords
            use_tools = self._auto_select_tools(task.query)
        
        print(f"     🔧 Using {len(use_tools)} analysis tools: {', '.join(use_tools)}")
        
        # Step 3: Run selected analysis tools
        tool_results = {}
        for tool_name in use_tools:
            if tool_name in self.tools:
                try:
                    tool_results[tool_name] = self.tools[tool_name](retrieved_data)
                    print(f"        ✅ {tool_name} complete")
                except Exception as e:
                    print(f"        ⚠️  {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
        
        # Step 4: Synthesize all results
        print(f"     🧠 Synthesizing comprehensive analysis...")
        synthesis = self._synthesize_analysis(task, retrieved_data, tool_results)
        
        print(f"     ✅ Analysis complete (confidence: {synthesis.get('confidence_level', 0.7)})")
        
        return {
            "agent": self.name,
            "task_id": task.id,
            "query": task.query,
            "retrieved_data": retrieved_data,
            "tool_analyses": tool_results,
            "synthesis": synthesis,
            "tools_used": use_tools,
            "timestamp": datetime.now().isoformat()
        }
    
    def _auto_select_tools(self, query: str) -> List[str]:
        """
        Automatically select appropriate tools based on query content
        """
        query_lower = query.lower()
        selected_tools = []
        
        # Comparative analysis keywords
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference", "better", "contrast"]):
            selected_tools.append("comparative_analysis")
        
        # Trend analysis keywords
        if any(word in query_lower for word in ["trend", "over time", "future", "forecast", "prediction", "emerging", "growth"]):
            selected_tools.append("trend_analysis")
        
        # Causal reasoning keywords
        if any(word in query_lower for word in ["why", "cause", "reason", "because", "due to", "effect", "impact", "result"]):
            selected_tools.append("causal_reasoning")
        
        # Statistical analysis keywords
        if any(word in query_lower for word in ["data", "statistics", "average", "correlation", "pattern", "distribution", "analyze"]):
            selected_tools.append("statistical_analysis")
        
        # Default: use all tools if no keywords matched
        if not selected_tools:
            selected_tools = list(self.tools.keys())
        
        return selected_tools
    
    def _synthesize_analysis(self, task: SubTask, retrieved_data: Dict[str, Any], 
                            tool_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize results from all tools into a comprehensive analysis
        """
        system_prompt = """You are synthesizing results from multiple analysis tools.

        Create a comprehensive analysis that:
        1. Integrates findings from all tools
        2. Identifies key insights across analyses
        3. Resolves any contradictions
        4. Provides actionable conclusions
        5. Assigns overall confidence level
        
        Return JSON:
        {
            "executive_summary": "High-level summary of all findings",
            "key_insights": ["insight1", "insight2", ...],
            "cross_tool_findings": [
                {
                    "finding": "finding description",
                    "supporting_tools": ["tool1", "tool2"],
                    "confidence": 0.0-1.0
                }
            ],
            "contradictions": ["any conflicting findings"],
            "conclusions": ["conclusion1", "conclusion2", ...],
            "recommendations": ["recommendation1", ...],
            "limitations": ["limitation1", ...],
            "confidence_level": 0.0-1.0,
            "next_steps": ["step1", "step2", ...]
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Task Query: {task.query}

Tool Analysis Results:
{json.dumps(tool_results, indent=2)}

Synthesize these analyses into comprehensive findings.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "executive_summary": response.content[:500],
                "key_insights": [],
                "conclusions": [],
                "confidence_level": 0.7
            }


# ============================================================================
# AGENT 3: FACT CHECKING & VALIDATION AGENT (WITH ADVANCED VALIDATION TOOLS)
# ============================================================================

class FactCheckingAgent:
    """
    Agent responsible for fact-checking and validating analysis results
    
    Enhanced with 4 specialized validation tools:
    1. Source Credibility Checker - Evaluates source reliability (uses web search)
    2. Cross-Reference Validator - Validates claims across multiple sources
    3. Confidence Score Calculator - Calculates weighted confidence scores
    4. Contradiction Detector - Identifies logical inconsistencies
    """
    
    def __init__(self, api_key: str = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
        self.name = "FactCheckingAgent"
        self.web_scraper = WebScraperAgent(api_key=api_key)
        
        # Initialize validation tools
        print(f"   🔧 Initializing {self.name} with 4 validation tools...")
        self._setup_validation_tools()
    
    def _setup_validation_tools(self):
        """Setup the four specialized validation tools"""
        self.tools = {
            "source_credibility": self._source_credibility_checker,
            "cross_reference": self._cross_reference_validator,
            "confidence_score": self._confidence_score_calculator,
            "contradiction_detector": self._contradiction_detector
        }
        print(f"      ✅ Source Credibility Checker loaded")
        print(f"      ✅ Cross-Reference Validator loaded")
        print(f"      ✅ Confidence Score Calculator loaded")
        print(f"      ✅ Contradiction Detector loaded")
    
    def _source_credibility_checker(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 1: Source Credibility Checker
        Evaluates the credibility and reliability of information sources
        Uses web search to verify source reputation
        
        Returns:
            - Credibility ratings for each source
            - Authority indicators
            - Bias assessment
            - Reliability score
        """
        print(f"        🔍 Running Source Credibility Check...")
        
        # Extract sources from data
        sources = []
        if isinstance(data.get('retrieved_data'), dict):
            sources_data = data['retrieved_data'].get('sources', [])
            if isinstance(sources_data, list):
                sources.extend(sources_data)
        
        # Search for source credibility information
        credibility_info = {}
        for source in sources[:5]:  # Limit to 5 sources to avoid excessive searches
            source_name = source if isinstance(source, str) else source.get('name', 'Unknown')
            print(f"           Checking credibility: {source_name[:50]}...")
            
            # Use web scraper to search for source reputation
            search_query = f"credibility reliability reputation {source_name}"
            search_results = self.web_scraper.retrieve_information(
                query=search_query,
                context="Source credibility verification"
            )
            credibility_info[source_name] = search_results
        
        # Analyze credibility using LLM
        system_prompt = """You are a Source Credibility expert specializing in evaluating information sources.

        Evaluate sources based on:
        1. Authority - Expertise and qualifications
        2. Accuracy - Track record of factual reporting
        3. Objectivity - Presence of bias or agenda
        4. Currency - Timeliness and up-to-date information
        5. Coverage - Depth and breadth of information
        
        Return JSON:
        {
            "overall_credibility": "high/medium/low",
            "source_ratings": [
                {
                    "source": "source name",
                    "credibility_score": 0.0-1.0,
                    "authority": "expert/moderate/limited",
                    "bias_level": "minimal/moderate/high",
                    "reliability": "very reliable/reliable/questionable/unreliable",
                    "indicators": {
                        "positive": ["indicator1", ...],
                        "negative": ["indicator1", ...]
                    },
                    "reputation": "well-established/emerging/unknown/controversial"
                }
            ],
            "red_flags": ["flag1", "flag2", ...],
            "verified_sources": ["source1", ...],
            "questionable_sources": ["source1", ...],
            "recommendations": ["recommendation1", ...],
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Evaluate source credibility:

Sources to Check:
{json.dumps(sources, indent=2)}

Credibility Research:
{json.dumps(credibility_info, indent=2)}
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "overall_credibility": "medium",
                "source_ratings": [],
                "red_flags": [],
                "confidence": 0.6
            }
    
    def _cross_reference_validator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 2: Cross-Reference Validator
        Validates claims by checking against multiple independent sources
        
        Returns:
            - Cross-referenced claims
            - Agreement levels
            - Conflicting information
            - Consensus findings
        """
        print(f"        🔗 Running Cross-Reference Validation...")
        
        # Extract key claims from analysis
        claims = []
        if 'synthesis' in data:
            claims.extend(data['synthesis'].get('key_insights', []))
            claims.extend(data['synthesis'].get('conclusions', []))
        elif 'analysis' in data:
            claims.extend(data['analysis'].get('insights', []))
            claims.extend(data['analysis'].get('conclusions', []))
        
        # Cross-reference top claims
        cross_ref_results = {}
        for claim in claims[:3]:  # Limit to top 3 claims
            if isinstance(claim, str) and len(claim) > 10:
                print(f"           Cross-referencing: {claim[:60]}...")
                
                # Search for supporting/contradicting information
                search_query = f"verify fact check {claim}"
                search_results = self.web_scraper.retrieve_information(
                    query=search_query,
                    context="Cross-reference validation"
                )
                cross_ref_results[claim] = search_results
        
        # Analyze cross-references
        system_prompt = """You are a Cross-Reference Validation expert.

        Validate claims by analyzing:
        1. Multiple source agreement - How many sources support the claim
        2. Source independence - Are sources truly independent
        3. Consistency - How consistent is the information
        4. Contradictions - Any conflicting information
        5. Consensus level - Degree of agreement
        
        Return JSON:
        {
            "cross_referenced_claims": [
                {
                    "claim": "the claim being validated",
                    "validation_status": "confirmed/partial/disputed/unverified",
                    "supporting_sources": 0-10,
                    "contradicting_sources": 0-10,
                    "agreement_level": 0.0-1.0,
                    "consistency": "highly consistent/consistent/inconsistent",
                    "evidence_quality": "strong/moderate/weak/insufficient",
                    "notes": "additional context"
                }
            ],
            "consensus_findings": [
                {
                    "finding": "consensus statement",
                    "support_level": 0.0-1.0,
                    "sources_count": 0
                }
            ],
            "conflicts": [
                {
                    "claim": "conflicting claim",
                    "sources": ["source1", "source2"],
                    "nature": "direct contradiction/partial conflict/contextual difference"
                }
            ],
            "unverifiable_claims": ["claim1", ...],
            "reliability_assessment": "high/medium/low",
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Cross-reference these claims:

Claims to Validate:
{json.dumps(claims, indent=2)}

Cross-Reference Research:
{json.dumps(cross_ref_results, indent=2)}
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "cross_referenced_claims": [],
                "consensus_findings": [],
                "conflicts": [],
                "reliability_assessment": "medium",
                "confidence": 0.6
            }
    
    def _confidence_score_calculator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 3: Confidence Score Calculator
        Calculates weighted confidence scores based on multiple factors
        
        Returns:
            - Overall confidence score
            - Factor-based breakdown
            - Uncertainty quantification
            - Confidence intervals
        """
        print(f"        📊 Calculating Confidence Scores...")
        
        system_prompt = """You are a Confidence Score Calculator expert.

        Calculate confidence based on:
        1. Source quality - Credibility and authority of sources
        2. Evidence strength - Quality and quantity of evidence
        3. Consistency - Agreement across sources
        4. Completeness - How comprehensive the information is
        5. Recency - How current the information is
        6. Methodology - Rigor of analysis methods
        
        Return JSON:
        {
            "overall_confidence": 0.0-1.0,
            "confidence_breakdown": {
                "source_quality": {
                    "score": 0.0-1.0,
                    "weight": 0.25,
                    "rationale": "explanation"
                },
                "evidence_strength": {
                    "score": 0.0-1.0,
                    "weight": 0.25,
                    "rationale": "explanation"
                },
                "consistency": {
                    "score": 0.0-1.0,
                    "weight": 0.20,
                    "rationale": "explanation"
                },
                "completeness": {
                    "score": 0.0-1.0,
                    "weight": 0.15,
                    "rationale": "explanation"
                },
                "recency": {
                    "score": 0.0-1.0,
                    "weight": 0.10,
                    "rationale": "explanation"
                },
                "methodology": {
                    "score": 0.0-1.0,
                    "weight": 0.05,
                    "rationale": "explanation"
                }
            },
            "confidence_interval": {
                "lower_bound": 0.0-1.0,
                "upper_bound": 0.0-1.0,
                "interpretation": "explanation"
            },
            "uncertainty_factors": [
                {
                    "factor": "uncertainty source",
                    "impact": "high/medium/low",
                    "description": "explanation"
                }
            ],
            "confidence_level": "very high/high/medium/low/very low",
            "reliability_grade": "A+/A/A-/B+/B/B-/C+/C/C-/D/F",
            "recommendations": ["recommendation1", ...]
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Calculate confidence scores for:

{json.dumps(data, indent=2)}
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "overall_confidence": 0.7,
                "confidence_breakdown": {},
                "confidence_level": "medium",
                "reliability_grade": "B"
            }
    
    def _contradiction_detector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 4: Contradiction Detector
        Identifies logical inconsistencies and contradictions
        
        Returns:
            - Detected contradictions
            - Severity assessment
            - Resolution recommendations
            - Logical errors
        """
        print(f"        ⚠️  Running Contradiction Detection...")
        
        system_prompt = """You are a Contradiction Detection expert specializing in logical analysis.

        Detect:
        1. Direct contradictions - Statements that directly oppose each other
        2. Logical inconsistencies - Reasoning that doesn't hold up
        3. Data conflicts - Conflicting data points or statistics
        4. Temporal inconsistencies - Timeline contradictions
        5. Scope contradictions - Conflicting claims about extent/magnitude
        
        Return JSON:
        {
            "contradictions_found": [
                {
                    "type": "direct/logical/data/temporal/scope",
                    "severity": "critical/major/minor",
                    "statement_1": "first contradicting statement",
                    "statement_2": "second contradicting statement",
                    "description": "explanation of contradiction",
                    "location": "where found (section/tool)",
                    "impact": "high/medium/low",
                    "resolution": "suggested resolution"
                }
            ],
            "logical_fallacies": [
                {
                    "fallacy_type": "type of fallacy",
                    "description": "explanation",
                    "location": "where found",
                    "correction": "suggested fix"
                }
            ],
            "inconsistencies": [
                {
                    "issue": "inconsistency description",
                    "severity": "high/medium/low",
                    "recommendation": "how to resolve"
                }
            ],
            "coherence_score": 0.0-1.0,
            "overall_assessment": "coherent/mostly coherent/some issues/major issues",
            "critical_issues": ["issue1", ...],
            "requires_revision": true/false,
            "confidence": 0.0-1.0
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Detect contradictions and logical inconsistencies in:

{json.dumps(data, indent=2)}
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "contradictions_found": [],
                "logical_fallacies": [],
                "inconsistencies": [],
                "coherence_score": 0.8,
                "overall_assessment": "mostly coherent",
                "requires_revision": False,
                "confidence": 0.7
            }
    
    def validate_analysis(self, analysis_result: Dict[str, Any], use_tools: List[str] = None) -> Dict[str, Any]:
        """
        Fact-checks and validates the analysis results using specialized tools
        
        Args:
            analysis_result: The analysis to validate
            use_tools: List of tool names to use (default: all tools)
                      Options: ["source_credibility", "cross_reference", 
                               "confidence_score", "contradiction_detector"]
        """
        print(f"  ✓ {self.name}: Validating analysis for task {analysis_result['task_id']}")
        
        # Step 1: Determine which tools to use
        if use_tools is None:
            use_tools = list(self.tools.keys())  # Use all tools by default
        
        print(f"     🔧 Using {len(use_tools)} validation tools: {', '.join(use_tools)}")
        
        # Step 2: Run validation tools
        tool_results = {}
        for tool_name in use_tools:
            if tool_name in self.tools:
                try:
                    tool_results[tool_name] = self.tools[tool_name](analysis_result)
                    print(f"        ✅ {tool_name} complete")
                except Exception as e:
                    print(f"        ⚠️  {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
        
        # Step 3: Synthesize validation results
        print(f"     🧠 Synthesizing validation results...")
        synthesis = self._synthesize_validation(analysis_result, tool_results)
        
        status = synthesis.get('validation_status', 'partial')
        score = synthesis.get('accuracy_score', 0.7)
        print(f"     ✅ Validation complete (status: {status}, score: {score})")
        
        return {
            "agent": self.name,
            "task_id": analysis_result['task_id'],
            "validation": synthesis,
            "tool_results": tool_results,
            "tools_used": use_tools,
            "original_analysis": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _synthesize_validation(self, analysis_result: Dict[str, Any], 
                               tool_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize results from all validation tools into final assessment
        """
        system_prompt = """You are synthesizing validation results from multiple tools.

        Create comprehensive validation assessment:
        1. Integrate findings from all validation tools
        2. Determine overall validation status
        3. Calculate final accuracy score
        4. Identify critical issues
        5. Provide recommendation (approve/revise/reject)
        
        Return JSON:
        {
            "validation_status": "verified/partial/flagged/rejected",
            "accuracy_score": 0.0-1.0,
            "verified_claims": ["claim1", "claim2", ...],
            "flagged_items": [
                {
                    "item": "flagged content",
                    "reason": "why it's flagged",
                    "severity": "critical/high/medium/low",
                    "source_tool": "which tool found it"
                }
            ],
            "critical_issues": ["issue1", ...],
            "inconsistencies": ["inconsistency1", ...],
            "source_quality_summary": "overall source quality assessment",
            "cross_reference_summary": "cross-validation findings summary",
            "confidence_adjustment": 0.0-1.0,
            "final_confidence": 0.0-1.0,
            "validation_notes": "comprehensive validation notes",
            "recommendation": "approve/revise_minor/revise_major/reject",
            "required_actions": ["action1", "action2", ...]
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Synthesize validation results:

Original Analysis:
{json.dumps(analysis_result.get('synthesis', {}), indent=2)[:1000]}

Validation Tool Results:
{json.dumps(tool_results, indent=2)}
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "validation_status": "partial",
                "accuracy_score": 0.7,
                "recommendation": "approve",
                "validation_notes": response.content[:500]
            }


# ============================================================================
# AGENT 4: OUTPUT FORMATTING AGENT (WITH ADVANCED FORMATTING TOOLS)
# ============================================================================

class OutputFormattingAgent:
    """
    Agent responsible for formatting final output as detailed research papers
    
    Enhanced with 4 specialized formatting tools:
    1. Report Structuring Tool - Creates research paper structure
    2. Citation Formatter - Formats citations and references
    3. Visualization Generator - Creates data visualizations and charts
    4. Executive Summary Generator - Generates comprehensive summaries
    
    Produces detailed research insights like academic research papers
    """
    
    def __init__(self, api_key: str = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
        self.name = "OutputFormattingAgent"
        
        # Initialize formatting tools
        print(f"   🔧 Initializing {self.name} with 4 formatting tools...")
        self._setup_formatting_tools()
    
    def _setup_formatting_tools(self):
        """Setup the four specialized formatting tools"""
        self.tools = {
            "report_structuring": self._report_structuring_tool,
            "citation_formatter": self._citation_formatter,
            "visualization_generator": self._visualization_generator,
            "executive_summary": self._executive_summary_generator
        }
        print(f"      ✅ Report Structuring Tool loaded")
        print(f"      ✅ Citation Formatter loaded")
        print(f"      ✅ Visualization Generator loaded")
        print(f"      ✅ Executive Summary Generator loaded")
    
    def _report_structuring_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 1: Report Structuring Tool
        Creates academic research paper structure with detailed sections
        
        Returns:
            - Abstract
            - Introduction with background
            - Literature Review
            - Methodology
            - Findings & Results
            - Discussion & Analysis
            - Conclusions
            - Recommendations
        """
        print(f"        📄 Structuring Research Report...")
        
        system_prompt = """You are a Research Report Structuring expert specializing in academic papers.

        Create a comprehensive research paper structure with:
        
        1. ABSTRACT (150-250 words)
           - Research objective
           - Methodology overview
           - Key findings
           - Main conclusions
        
        2. INTRODUCTION
           - Background and context
           - Research problem statement
           - Research questions
           - Scope and limitations
           - Significance of research
        
        3. LITERATURE REVIEW
           - Current state of knowledge
           - Key theories and frameworks
           - Previous studies and findings
           - Research gaps identified
        
        4. METHODOLOGY
           - Research design
           - Data collection methods
           - Analysis tools and techniques
           - Validation approach
        
        5. FINDINGS & RESULTS
           - Primary findings (detailed)
           - Secondary findings
           - Data patterns and trends
           - Statistical results
        
        6. DISCUSSION & ANALYSIS
           - Interpretation of findings
           - Comparison with literature
           - Implications
           - Limitations
        
        7. CONCLUSIONS
           - Summary of key findings
           - Theoretical contributions
           - Practical implications
        
        8. RECOMMENDATIONS
           - Actionable recommendations
           - Future research directions
        
        Return JSON with detailed content for each section:
        {
            "title": "Research paper title",
            "abstract": {
                "objective": "research objective",
                "methodology": "methods used",
                "key_findings": ["finding1", "finding2", ...],
                "conclusions": "main conclusion",
                "word_count": 200
            },
            "introduction": {
                "background": "detailed background (500+ words)",
                "problem_statement": "clear problem definition",
                "research_questions": ["question1", "question2", ...],
                "objectives": ["objective1", ...],
                "scope": "research scope",
                "limitations": ["limitation1", ...],
                "significance": "why this research matters"
            },
            "literature_review": {
                "current_knowledge": "state of the field (300+ words)",
                "key_theories": ["theory1", "theory2", ...],
                "previous_studies": [
                    {
                        "study": "study description",
                        "findings": "what they found",
                        "relevance": "how it relates"
                    }
                ],
                "research_gaps": ["gap1", "gap2", ...]
            },
            "methodology": {
                "research_design": "design description",
                "data_collection": {
                    "methods": ["method1", "method2", ...],
                    "sources": ["source1", ...],
                    "tools": ["tool1", ...]
                },
                "analysis_techniques": ["technique1", ...],
                "validation_approach": "how validated",
                "limitations": ["methodological limitation1", ...]
            },
            "findings": {
                "overview": "summary of findings",
                "primary_findings": [
                    {
                        "finding": "detailed finding description (100+ words)",
                        "evidence": "supporting evidence",
                        "significance": "why it matters",
                        "confidence": 0.0-1.0
                    }
                ],
                "secondary_findings": ["finding1", ...],
                "patterns": ["pattern1", ...],
                "trends": ["trend1", ...],
                "statistical_summary": {}
            },
            "discussion": {
                "interpretation": "detailed interpretation (400+ words)",
                "comparison_with_literature": "how findings compare",
                "theoretical_implications": ["implication1", ...],
                "practical_implications": ["implication1", ...],
                "unexpected_findings": ["finding1", ...],
                "limitations": ["limitation1", ...]
            },
            "conclusions": {
                "summary": "comprehensive summary (300+ words)",
                "key_contributions": ["contribution1", ...],
                "theoretical_contributions": ["contribution1", ...],
                "practical_contributions": ["contribution1", ...],
                "final_remarks": "closing thoughts"
            },
            "recommendations": {
                "immediate_actions": [
                    {
                        "action": "what to do",
                        "priority": "high/medium/low",
                        "rationale": "why do it",
                        "expected_impact": "anticipated outcome"
                    }
                ],
                "future_research": ["research direction1", ...],
                "implementation_guidelines": ["guideline1", ...]
            }
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Structure this data into a detailed research paper:

{json.dumps(data, indent=2)[:3000]}

Create comprehensive, detailed sections with substantial content.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "title": "Research Report",
                "abstract": {"objective": "Research analysis", "key_findings": []},
                "introduction": {"background": response.content[:500]},
                "error": "Failed to parse full structure"
            }
    
    def _citation_formatter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 2: Citation Formatter
        Formats citations and creates bibliography in multiple styles
        
        Returns:
            - In-text citations
            - Reference list (APA, MLA, Chicago)
            - Source metadata
        """
        print(f"        📚 Formatting Citations...")
        
        system_prompt = """You are a Citation Formatting expert.

        Format citations in multiple academic styles:
        
        1. Extract all sources from the data
        2. Create in-text citations
        3. Generate reference list in APA, MLA, and Chicago styles
        4. Number citations appropriately
        5. Include DOIs, URLs, and access dates
        
        Return JSON:
        {
            "sources_extracted": [
                {
                    "source_id": "source_1",
                    "title": "source title",
                    "author": "author(s)",
                    "date": "publication date",
                    "url": "URL if available",
                    "type": "journal/website/book/report"
                }
            ],
            "in_text_citations": {
                "apa": ["(Author, 2024)", "(Smith & Jones, 2023)", ...],
                "mla": ["(Author 123)", "(Smith and Jones 45)", ...],
                "chicago": ["(Author 2024, 123)", ...]
            },
            "references": {
                "apa": [
                    "Author, A. (2024). Title of work. Publisher. https://doi.org/..."
                ],
                "mla": [
                    "Author, A. Title of Work. Publisher, 2024."
                ],
                "chicago": [
                    "Author, A. 2024. Title of Work. Publisher."
                ],
                "ieee": [
                    "[1] A. Author, \"Title,\" Journal, vol. 1, pp. 1-10, 2024."
                ]
            },
            "citation_count": 0,
            "unique_sources": 0,
            "source_types": {
                "journals": 0,
                "websites": 0,
                "books": 0,
                "reports": 0
            },
            "citation_map": {
                "finding_1": ["source_1", "source_3"],
                "finding_2": ["source_2", "source_4"]
            }
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Extract and format citations from:

{json.dumps(data, indent=2)[:2000]}

Create comprehensive citations in multiple styles.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "sources_extracted": [],
                "references": {"apa": [], "mla": [], "chicago": []},
                "citation_count": 0
            }
    
    def _visualization_generator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 3: Visualization Generator
        Generates data visualization descriptions and chart specifications
        
        Returns:
            - Chart specifications
            - Graph descriptions
            - Data tables
            - Visual summaries
        """
        print(f"        📊 Generating Visualizations...")
        
        system_prompt = """You are a Data Visualization expert.

        Generate visualization specifications for research data:
        
        1. Identify data suitable for visualization
        2. Recommend appropriate chart types
        3. Create chart specifications
        4. Design data tables
        5. Generate visual descriptions
        
        Return JSON:
        {
            "visualizations": [
                {
                    "viz_id": "viz_1",
                    "title": "Chart Title",
                    "type": "bar/line/pie/scatter/heatmap/network/timeline",
                    "description": "What this visualization shows",
                    "data_summary": {
                        "x_axis": "variable name",
                        "y_axis": "variable name",
                        "categories": ["cat1", "cat2", ...],
                        "values": [1, 2, 3, ...]
                    },
                    "chart_spec": {
                        "chart_type": "specific chart type",
                        "title": "chart title",
                        "x_label": "x-axis label",
                        "y_label": "y-axis label",
                        "data_points": [
                            {"label": "A", "value": 10},
                            {"label": "B", "value": 20}
                        ],
                        "colors": ["#color1", "#color2"],
                        "annotations": ["note1", ...]
                    },
                    "insights": "key insights from this visualization",
                    "placement": "where in paper (e.g., Section 4.1)"
                }
            ],
            "data_tables": [
                {
                    "table_id": "table_1",
                    "title": "Table Title",
                    "caption": "Table caption",
                    "headers": ["Column1", "Column2", ...],
                    "rows": [
                        ["value1", "value2", ...],
                        ["value1", "value2", ...]
                    ],
                    "notes": ["note1", ...],
                    "placement": "where in paper"
                }
            ],
            "figures": [
                {
                    "figure_id": "fig_1",
                    "title": "Figure Title",
                    "type": "diagram/flowchart/model/framework",
                    "description": "detailed description (300+ words)",
                    "components": ["component1", "component2", ...],
                    "relationships": ["relationship1", ...],
                    "caption": "figure caption"
                }
            ],
            "infographics": [
                {
                    "title": "Infographic Title",
                    "sections": [
                        {
                            "heading": "section heading",
                            "content": "content",
                            "visual_elements": ["element1", ...]
                        }
                    ]
                }
            ],
            "visualization_summary": "overview of all visualizations"
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Generate visualization specifications for:

{json.dumps(data, indent=2)[:2000]}

Create comprehensive visual representations of the data.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "visualizations": [],
                "data_tables": [],
                "figures": [],
                "visualization_summary": "Visualization generation in progress"
            }
    
    def _executive_summary_generator(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 4: Executive Summary Generator
        Creates comprehensive multi-level summaries
        
        Returns:
            - One-page executive summary
            - Extended summary
            - Key highlights
            - Strategic insights
        """
        print(f"        📋 Generating Executive Summary...")
        
        system_prompt = """You are an Executive Summary expert.

        Create multi-level summaries:
        
        1. ELEVATOR PITCH (30 seconds - 50 words)
        2. ONE-PAGE SUMMARY (1 page - 300-400 words)
        3. EXTENDED SUMMARY (2-3 pages - 600-900 words)
        4. KEY HIGHLIGHTS (bullet points)
        5. STRATEGIC INSIGHTS (decision-maker focus)
        
        Return JSON:
        {
            "elevator_pitch": {
                "text": "30-second summary (50 words)",
                "word_count": 50
            },
            "one_page_summary": {
                "overview": "Research overview (100 words)",
                "key_findings": "Main findings (100 words)",
                "implications": "What it means (50 words)",
                "recommendations": "What to do (50 words)",
                "word_count": 300
            },
            "extended_summary": {
                "context": "Background and context (200 words)",
                "methodology": "How research was done (100 words)",
                "findings": "Detailed findings (300 words)",
                "analysis": "Analysis and interpretation (200 words)",
                "conclusions": "Conclusions (100 words)",
                "next_steps": "Recommended actions (100 words)",
                "word_count": 900
            },
            "key_highlights": [
                {
                    "category": "highlight category",
                    "points": ["point1", "point2", ...],
                    "significance": "why it matters"
                }
            ],
            "strategic_insights": {
                "opportunities": [
                    {
                        "opportunity": "description",
                        "potential_impact": "high/medium/low",
                        "timeframe": "immediate/short-term/long-term"
                    }
                ],
                "risks": [
                    {
                        "risk": "description",
                        "severity": "high/medium/low",
                        "mitigation": "how to address"
                    }
                ],
                "decision_points": [
                    {
                        "decision": "what needs to be decided",
                        "options": ["option1", "option2"],
                        "recommendation": "recommended choice",
                        "rationale": "why"
                    }
                ]
            },
            "talking_points": [
                "point1 (for presentations)",
                "point2",
                "point3"
            ],
            "quick_facts": [
                "fact1 (key statistics)",
                "fact2",
                "fact3"
            ]
        }"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Generate comprehensive executive summaries for:

{json.dumps(data, indent=2)[:2000]}

Create summaries at multiple levels of detail.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "elevator_pitch": {"text": response.content[:200], "word_count": 50},
                "one_page_summary": {"overview": response.content[:500]},
                "key_highlights": []
            }
    
    def format_output(self, synthesized_results: Dict[str, Any], 
                     use_tools: List[str] = None,
                     output_style: str = "concise",
                     page_limit: int = 3,
                     include_visualizations: bool = False) -> Dict[str, Any]:
        """
        Formats the synthesized results into a structured report
        
        Args:
            synthesized_results: Results to format
            use_tools: List of tool names to use (default: based on output_style)
                      Options: ["report_structuring", "citation_formatter", 
                               "visualization_generator", "executive_summary"]
            output_style: Output format style
                         - "concise" (default): 2-3 page structured report
                         - "research_paper": 15-20 page detailed research paper
                         - "standard": Brief summary report
            page_limit: Maximum number of pages (1-20)
                       - 1-3 pages: Concise report
                       - 4-7 pages: Medium report
                       - 8-20 pages: Full research paper
            include_visualizations: Whether to include visualizations (default: False)
        """
        print(f"  📝 {self.name}: Formatting output as {output_style} ({page_limit} pages max)...")
        
        # Step 1: Determine which tools to use based on output style
        if use_tools is None:
            if output_style == "concise":
                # For concise reports: skip visualization tool
                use_tools = ["report_structuring", "citation_formatter", "executive_summary"]
            elif output_style == "research_paper":
                # For research papers: use all tools if visualizations enabled
                use_tools = list(self.tools.keys()) if include_visualizations else \
                           ["report_structuring", "citation_formatter", "executive_summary"]
            else:  # standard
                use_tools = ["executive_summary"]
        
        # Remove visualization tool if not requested
        if not include_visualizations and "visualization_generator" in use_tools:
            use_tools.remove("visualization_generator")
        
        print(f"     🔧 Using {len(use_tools)} formatting tools: {', '.join(use_tools)}")
        if not include_visualizations:
            print(f"     📊 Visualizations: Disabled (text-only report)")
        
        # Step 2: Run formatting tools
        tool_results = {}
        for tool_name in use_tools:
            if tool_name in self.tools:
                try:
                    tool_results[tool_name] = self.tools[tool_name](synthesized_results)
                    print(f"        ✅ {tool_name} complete")
                except Exception as e:
                    print(f"        ⚠️  {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
        
        # Step 3: Synthesize into final formatted output based on style and page limit
        print(f"     🧠 Synthesizing final report...")
        
        if output_style == "concise" or page_limit <= 3:
            final_output = self._create_concise_report(synthesized_results, tool_results, page_limit)
        elif output_style == "research_paper" or page_limit >= 8:
            final_output = self._create_research_paper(synthesized_results, tool_results, page_limit, include_visualizations)
        else:  # medium or standard
            final_output = self._create_medium_report(synthesized_results, tool_results, page_limit)
        
        print(f"     ✅ Formatting complete - Generated {page_limit}-page {output_style} report")
        
        return {
            "agent": self.name,
            "output_style": output_style,
            "page_limit": page_limit,
            "include_visualizations": include_visualizations,
            "formatted_output": final_output,
            "tool_results": tool_results,
            "tools_used": use_tools,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_concise_report(self, data: Dict[str, Any], 
                               tool_results: Dict[str, Dict[str, Any]],
                               page_limit: int = 3) -> Dict[str, Any]:
        """
        Creates a concise 2-3 page report (no visualizations)
        Focused on key findings, analysis, and recommendations
        """
        report_structure = tool_results.get('report_structuring', {})
        citations = tool_results.get('citation_formatter', {})
        executive_summary = tool_results.get('executive_summary', {})
        
        # Calculate word count target (500 words per page)
        word_count_target = page_limit * 500
        
        # Build concise report
        concise_report = {
            "document_type": "Concise Research Report",
            "metadata": {
                "generated_date": datetime.now().strftime("%B %d, %Y"),
                "version": "1.0",
                "page_limit": f"{page_limit} pages",
                "word_count_target": f"{word_count_target} words",
                "format": "Text-only (no visualizations)"
            },
            
            # Title and Summary
            "title": report_structure.get('title', 'Research Analysis Report'),
            
            # Executive Summary (condensed)
            "executive_summary": {
                "overview": executive_summary.get('one_page_summary', {}).get('overview', 'N/A'),
                "key_findings": executive_summary.get('one_page_summary', {}).get('key_findings', 'N/A'),
                "recommendations": executive_summary.get('one_page_summary', {}).get('recommendations', 'N/A')
            },
            
            # Key Highlights
            "key_highlights": executive_summary.get('key_highlights', [])[:5],  # Limit to 5
            
            # Quick Facts
            "quick_facts": executive_summary.get('quick_facts', [])[:5],  # Limit to 5
            
            # Main Sections (condensed)
            "introduction": {
                "background": report_structure.get('introduction', {}).get('background', 'N/A')[:400],
                "research_questions": report_structure.get('introduction', {}).get('research_questions', [])[:3],
                "scope": report_structure.get('introduction', {}).get('scope', 'N/A')[:200]
            },
            
            "methodology": {
                "approach": report_structure.get('methodology', {}).get('research_design', 'N/A')[:300],
                "data_sources": report_structure.get('methodology', {}).get('data_collection', {}).get('sources', [])[:5]
            },
            
            "findings": {
                "summary": report_structure.get('findings', {}).get('overview', 'N/A')[:400],
                "primary_findings": [
                    {
                        "finding": f.get('finding', 'N/A')[:200],
                        "significance": f.get('significance', 'N/A')[:100],
                        "confidence": f.get('confidence', 0.8)
                    }
                    for f in report_structure.get('findings', {}).get('primary_findings', [])[:5]
                ],
                "key_patterns": report_structure.get('findings', {}).get('patterns', [])[:3]
            },
            
            "discussion": {
                "analysis": report_structure.get('discussion', {}).get('interpretation', 'N/A')[:400],
                "implications": report_structure.get('discussion', {}).get('practical_implications', [])[:5],
                "limitations": report_structure.get('discussion', {}).get('limitations', [])[:3]
            },
            
            "conclusions": {
                "summary": report_structure.get('conclusions', {}).get('summary', 'N/A')[:300],
                "key_contributions": report_structure.get('conclusions', {}).get('key_contributions', [])[:3]
            },
            
            "recommendations": {
                "immediate_actions": [
                    {
                        "action": r.get('action', 'N/A'),
                        "priority": r.get('priority', 'medium'),
                        "rationale": r.get('rationale', 'N/A')[:150]
                    }
                    for r in report_structure.get('recommendations', {}).get('immediate_actions', [])[:5]
                ],
                "future_directions": report_structure.get('recommendations', {}).get('future_research', [])[:3]
            },
            
            # Strategic Insights (condensed)
            "strategic_insights": {
                "opportunities": [
                    {
                        "opportunity": o.get('opportunity', 'N/A')[:150],
                        "impact": o.get('potential_impact', 'medium'),
                        "timeframe": o.get('timeframe', 'short-term')
                    }
                    for o in executive_summary.get('strategic_insights', {}).get('opportunities', [])[:3]
                ],
                "risks": [
                    {
                        "risk": r.get('risk', 'N/A')[:150],
                        "severity": r.get('severity', 'medium'),
                        "mitigation": r.get('mitigation', 'N/A')[:100]
                    }
                    for r in executive_summary.get('strategic_insights', {}).get('risks', [])[:3]
                ]
            },
            
            # References (simplified)
            "references": {
                "citation_style": "APA",
                "key_sources": citations.get('references', {}).get('apa', [])[:10],
                "total_sources": citations.get('citation_count', 0)
            },
            
            # Footer
            "report_metrics": {
                "estimated_pages": page_limit,
                "sections_included": 7,
                "visualizations_included": 0,
                "format": "Concise text-only report"
            }
        }
        
        return concise_report
    
    def _create_medium_report(self, data: Dict[str, Any], 
                             tool_results: Dict[str, Dict[str, Any]],
                             page_limit: int = 5) -> Dict[str, Any]:
        """
        Creates a medium 4-7 page report
        More detail than concise, less than full research paper
        """
        report_structure = tool_results.get('report_structuring', {})
        citations = tool_results.get('citation_formatter', {})
        executive_summary = tool_results.get('executive_summary', {})
        
        # Word count target
        word_count_target = page_limit * 500
        
        medium_report = {
            "document_type": "Research Report",
            "metadata": {
                "generated_date": datetime.now().strftime("%B %d, %Y"),
                "version": "1.0",
                "page_limit": f"{page_limit} pages",
                "word_count_target": f"{word_count_target} words"
            },
            
            "title": report_structure.get('title', 'Research Analysis Report'),
            "abstract": report_structure.get('abstract', {}),
            
            # Executive summaries
            "executive_summary": executive_summary.get('one_page_summary', {}),
            "key_highlights": executive_summary.get('key_highlights', [])[:8],
            
            # Main sections (moderate detail)
            "introduction": report_structure.get('introduction', {}),
            "methodology": report_structure.get('methodology', {}),
            
            "findings": {
                "overview": report_structure.get('findings', {}).get('overview', 'N/A'),
                "primary_findings": report_structure.get('findings', {}).get('primary_findings', [])[:8],
                "secondary_findings": report_structure.get('findings', {}).get('secondary_findings', [])[:5],
                "patterns": report_structure.get('findings', {}).get('patterns', [])[:5]
            },
            
            "discussion": report_structure.get('discussion', {}),
            "conclusions": report_structure.get('conclusions', {}),
            "recommendations": report_structure.get('recommendations', {}),
            
            # Strategic insights
            "strategic_insights": executive_summary.get('strategic_insights', {}),
            
            # Citations
            "references": citations.get('references', {}),
            "citation_count": citations.get('citation_count', 0),
            
            "report_metrics": {
                "estimated_pages": page_limit,
                "format": "Medium-detail research report"
            }
        }
        
        return medium_report
    
    def _create_research_paper(self, data: Dict[str, Any], 
                               tool_results: Dict[str, Dict[str, Any]],
                               page_limit: int = 20,
                               include_visualizations: bool = False) -> Dict[str, Any]:
        """
        Creates a detailed research paper format (8-20 pages)
        """
        # Get structured report
        report_structure = tool_results.get('report_structuring', {})
        citations = tool_results.get('citation_formatter', {})
        visualizations = tool_results.get('visualization_generator', {}) if include_visualizations else {}
        executive_summary = tool_results.get('executive_summary', {})
        
        # Word count target
        word_count_target = page_limit * 500
        
        # Build comprehensive research paper
        research_paper = {
            "document_type": "Research Paper",
            "metadata": {
                "generated_date": datetime.now().strftime("%B %d, %Y"),
                "version": "1.0",
                "page_limit": f"{page_limit} pages",
                "word_count_target": f"{word_count_target} words",
                "visualizations": "Included" if include_visualizations else "Not included"
            },
            
            # Front Matter
            "title": report_structure.get('title', 'Research Analysis Report'),
            "abstract": report_structure.get('abstract', {}),
            "keywords": self._extract_keywords(data),
            
            # Executive Summaries (Multiple Levels)
            "executive_summaries": executive_summary,
            
            # Main Content
            "introduction": report_structure.get('introduction', {}),
            "literature_review": report_structure.get('literature_review', {}),
            "methodology": report_structure.get('methodology', {}),
            "findings": report_structure.get('findings', {}),
            "discussion": report_structure.get('discussion', {}),
            "conclusions": report_structure.get('conclusions', {}),
            "recommendations": report_structure.get('recommendations', {}),
            
            # Supporting Materials
            "citations": citations,
            
            # Appendices
            "appendices": {
                "appendix_a": {
                    "title": "Additional Data",
                    "content": self._create_raw_data_summary(data)
                }
            },
            
            # References
            "references": citations.get('references', {}),
            "bibliography_count": citations.get('citation_count', 0)
        }
        
        # Add visualizations only if requested
        if include_visualizations and visualizations:
            research_paper["visualizations"] = visualizations
            research_paper["appendices"]["appendix_b"] = {
                "title": "Figures and Charts",
                "content": visualizations.get('figures', [])
            }
        
        return research_paper
    
    def _create_standard_output(self, data: Dict[str, Any], 
                                tool_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Creates a standard brief output format
        """
        executive_summary = tool_results.get('executive_summary', {})
        
        return {
            "document_type": "Summary Report",
            "executive_summary": executive_summary.get('one_page_summary', {}),
            "key_highlights": executive_summary.get('key_highlights', []),
            "quick_facts": executive_summary.get('quick_facts', []),
            "recommendations": data.get('recommendations', []),
            "next_steps": data.get('next_steps', [])
        }
    
    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        """Extract keywords from the data"""
        keywords = []
        
        # Extract from synthesis if available
        if 'synthesis' in data:
            synthesis = data['synthesis']
            if 'key_insights' in synthesis:
                keywords.extend([str(insight)[:50] for insight in synthesis['key_insights'][:5]])
        
        return keywords if keywords else ["research", "analysis", "findings"]
    
    def _create_raw_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of raw data"""
        return {
            "total_tasks_analyzed": len(data.get('task_results', [])),
            "total_sources_consulted": "Calculated from web scraper",
            "analysis_tools_used": data.get('tools_used', []),
            "validation_methods": "Multi-tool validation pipeline",
            "data_collection_period": datetime.now().strftime("%B %Y")
        }


# ============================================================================
# MANAGER AGENT (RESEARCH ORCHESTRATOR)
# ============================================================================

class ManagerAgent:
    """
    Main orchestrator that manages the entire research workflow
    Coordinates all sub-agents
    """
    
    def __init__(self, api_key: str = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
        self.name = "ManagerAgent"
        
        # Initialize all sub-agents
        self.deep_analysis_agent = DeepAnalysisAgent(api_key=api_key)
        self.fact_checking_agent = FactCheckingAgent(api_key=api_key)
        self.output_formatting_agent = OutputFormattingAgent(api_key=api_key)
        
        # Note: WebScraperAgent is used by DeepAnalysisAgent
    
    def decompose_query(self, query: str) -> List[SubTask]:
        """
        Breaks down the main query into sub-tasks
        """
        print(f"\n📋 {self.name}: Decomposing query into sub-tasks...")
        
        system_prompt = """You are a Manager Agent responsible for breaking down
        complex research queries into focused sub-tasks.
        
        Return your response as a JSON array of tasks:
        [
            {
                "id": "task_1",
                "query": "specific sub-query",
                "priority": 1-5
            }
        ]
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Break down this query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            tasks_data = json.loads(content)
            tasks = [SubTask(**task) for task in tasks_data]
        except Exception as e:
            # Fallback: create basic tasks
            tasks = [
                SubTask(id="task_1", query=query, priority=1)
            ]
        
        print(f"   ✅ Created {len(tasks)} sub-tasks")
        return tasks
    
    def process_task(self, task: SubTask) -> SubTask:
        """
        Process a single task through the agent pipeline:
        1. Deep Analysis (which uses Web Scraper)
        2. Fact Checking
        """
        print(f"\n🔄 {self.name}: Processing task {task.id}")
        print(f"   Query: {task.query}")
        
        # Step 1: Deep Analysis (includes web scraping)
        task.status = "analyzing"
        analysis_result = self.deep_analysis_agent.analyze_task(task)
        task.deep_analysis_result = analysis_result
        
        # Step 2: Fact Checking
        task.status = "fact_checking"
        fact_check_result = self.fact_checking_agent.validate_analysis(analysis_result)
        task.fact_check_result = fact_check_result
        
        # Update status
        task.status = "completed"
        
        return task
    
    def synthesize_results(self, completed_tasks: List[SubTask]) -> Dict[str, Any]:
        """
        Synthesize results from all completed tasks
        """
        print(f"\n🔮 {self.name}: Synthesizing results from {len(completed_tasks)} tasks...")
        
        system_prompt = """You are a Manager Agent synthesizing research findings.
        
        Combine insights from multiple tasks into a coherent whole.
        
        Return synthesis as JSON:
        {
            "overall_summary": "Comprehensive summary",
            "combined_insights": ["insight1", "insight2", ...],
            "key_findings": ["finding1", "finding2", ...],
            "cross_task_patterns": ["pattern1", ...],
            "synthesized_recommendations": ["rec1", "rec2", ...],
            "confidence_score": 0.0-1.0,
            "task_summaries": []
        }
        """
        
        # Prepare task data
        tasks_data = []
        for task in completed_tasks:
            tasks_data.append({
                "id": task.id,
                "query": task.query,
                "priority": task.priority,
                "analysis": task.deep_analysis_result,
                "validation": task.fact_check_result
            })
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Synthesize these task results:\n{json.dumps(tasks_data, indent=2)}")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            synthesis = json.loads(content)
        except:
            synthesis = {
                "overall_summary": response.content,
                "combined_insights": [],
                "key_findings": []
            }
        
        print(f"   ✅ Synthesis complete")
        
        return synthesis
    
    def orchestrate_research(self, query: str, 
                           page_limit: int = 3, 
                           include_visualizations: bool = False) -> Dict[str, Any]:
        """
        Main orchestration method that runs the complete workflow
        
        Args:
            query: The research query
            page_limit: Maximum number of pages for output (1-20)
                       - 1-3: Concise report (default)
                       - 4-7: Medium report
                       - 8-20: Full research paper
            include_visualizations: Whether to include charts and visualizations (default: False)
        """
        print("\n" + "="*80)
        print(f"🤖 MULTI-AGENT RESEARCH SYSTEM")
        print("="*80)
        print(f"\n📝 Original Query: {query}")
        print(f"📄 Page Limit: {page_limit} pages")
        print(f"📊 Visualizations: {'Enabled' if include_visualizations else 'Disabled'}\n")
        
        # Step 1: Decompose query into sub-tasks
        tasks = self.decompose_query(query)
        
        # Step 2: Process each task through the pipeline
        completed_tasks = []
        for i, task in enumerate(tasks, 1):
            print(f"\n{'─'*80}")
            print(f"Processing Task {i}/{len(tasks)}")
            print(f"{'─'*80}")
            completed_task = self.process_task(task)
            completed_tasks.append(completed_task)
        
        # Step 3: Synthesize all results
        print(f"\n{'─'*80}")
        print("Final Synthesis")
        print(f"{'─'*80}")
        synthesized_results = self.synthesize_results(completed_tasks)
        
        # Step 4: Format output with page limit and visualization settings
        final_output = self.output_formatting_agent.format_output(
            synthesized_results,
            page_limit=page_limit,
            include_visualizations=include_visualizations
        )
        
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETE")
        print("="*80 + "\n")
        
        return {
            "original_query": query,
            "page_limit": page_limit,
            "visualizations_included": include_visualizations,
            "tasks_processed": len(completed_tasks),
            "completed_tasks": [
                {
                    "id": t.id,
                    "query": t.query,
                    "status": t.status
                } for t in completed_tasks
            ],
            "synthesis": synthesized_results,
            "formatted_output": final_output.get('formatted_output', {}),
            "output_metadata": {
                "output_style": final_output.get('output_style', 'concise'),
                "page_limit": final_output.get('page_limit', page_limit),
                "tools_used": final_output.get('tools_used', [])
            },
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_multi_agent_system(api_key: str = None) -> ManagerAgent:
    """
    Factory function to create the multi-agent system
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        ManagerAgent instance that coordinates all sub-agents
    """
    return ManagerAgent(api_key=api_key)


def run_research(query: str, api_key: str = None, 
                page_limit: int = 3, 
                include_visualizations: bool = False,
                save_format: str = None,
                output_filename: str = None) -> Dict[str, Any]:
    """
    Convenience function to run a complete research workflow
    
    Args:
        query: The research query
        api_key: OpenAI API key
        page_limit: Maximum number of pages (1-20, default: 3)
        include_visualizations: Whether to include visualizations (default: False)
        save_format: Output format - 'txt', 'pdf', 'both', or None for JSON only (default: None)
        output_filename: Custom filename for output (default: 'research_report')
        
    Returns:
        Complete research results
    """
    manager = create_multi_agent_system(api_key=api_key)
    results = manager.orchestrate_research(query, page_limit, include_visualizations)
    
    # Save to text/PDF if requested
    if save_format:
        from text_report_generator import save_research_report
        
        filename = output_filename or "research_report"
        formats = []
        
        if save_format == 'both':
            formats = ['txt', 'pdf']
        elif save_format in ['txt', 'pdf']:
            formats = [save_format]
        
        if formats:
            word_target = page_limit * 500  # ~500 words per page
            saved_files = save_research_report(results, filename, formats, word_target)
            results['saved_files'] = saved_files
    
    return results

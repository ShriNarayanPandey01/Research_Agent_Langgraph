"""
Multi-Agent Research System with LangGraph
This implements a sophisticated multi-agent system using LangGraph with proper workflow:

WORKFLOW:
1. User Query -> Research Coordinator (Query Decomposition & Task Management)
2. For each subtask:
   - Research Coordinator sets current task
   - Web Scraper Agent retrieves information
   - Deep Analysis Agent analyzes (with 4 analysis tools)
   - Fact Checker Agent validates (with 4 validation tools)
   - Task Complete Handler increments task index
   - Research Coordinator checks for more tasks
3. Research Coordinator triggers synthesis when all tasks complete
4. Research Coordinator -> Output Formatter Agent
5. Final Report Generation

Agents:
- Research Coordinator Agent (orchestrator - handles decomposition, task routing, and completion checking)
- Deep Analysis Agent (with 4 analysis tools)
- Fact Checker Agent (with 4 validation tools)
- Output Formatter Agent (with 3 formatting tools)
- Web Scraper Agent (NOT react agent - just retrieval with RAG caching)
"""

from typing import TypedDict, List, Dict, Any, Annotated, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
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
from dotenv import load_dotenv  

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class ResearchState(TypedDict):
    """State for the research workflow"""
    # Input
    original_query: str
    page_limit: int
    include_visualizations: bool
    
    # Query Decomposition
    subtasks: List[Dict[str, Any]]
    current_task_index: int
    
    # Task Processing
    current_task: Dict[str, Any]
    web_data: Dict[str, Any]
    analysis_results: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    
    # Synthesis
    synthesized_result: Dict[str, Any]
    
    # Output
    final_output: Dict[str, Any]
    
    # Metadata
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str


# ============================================================================
# WEB SCRAPER AGENT (NOT REACT - JUST RETRIEVAL)
# ============================================================================

class WebScraperAgent:
    """
    Web Scraper Agent for fetching information
    NOT a ReAct agent - just a retrieval system with RAG caching
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = ".web_scraper_cache", 
                 use_pgvector: bool = True, pg_connection_string: str = None):
        # Initialize LLM - only pass api_key if provided, otherwise use env var
        if api_key:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
            self.openai_client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        else:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            self.openai_client = OpenAI()
            self.embeddings = OpenAIEmbeddings()
        
        self.name = "WebScraperAgent"
        
        # Initialize pgvector RAG if enabled
        self.use_pgvector = use_pgvector
        self.vector_store = None
        
        if use_pgvector:
            if pg_connection_string is None:
                pg_connection_string = "postgresql+psycopg2://shri:shri123@localhost:6024/vectordb"
            
            try:
                self.vector_store = PGVectorStore(
                    embeddings=self.embeddings,
                    collection_name="research_cache",
                    connection=pg_connection_string,
                    use_jsonb=True,
                )
                print(f"   📚 pgvector RAG initialized")
            except Exception as e:
                print(f"   ⚠️  pgvector initialization failed: {e}")
                self.use_pgvector = False
        
        # Fallback: File-based cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, Dict]:
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_query_hash(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_from_cache(self, query_hash: str, query: str = "") -> Dict[str, Any]:
        # Check memory
        if query_hash in self._memory_cache:
            print(f"     💾 Cache HIT (memory)")
            return self._memory_cache[query_hash]
        
        # Check pgvector
        if self.use_pgvector and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=1, filter={"query_hash": query_hash})
                if docs and len(docs) > 0:
                    cached_data = json.loads(docs[0].page_content)
                    self._memory_cache[query_hash] = cached_data
                    print(f"     💾 Cache HIT (pgvector)")
                    return cached_data
            except Exception as e:
                pass
        
        # Check disk
        if query_hash in self.cache_index:
            cache_file = self.cache_dir / f"{query_hash}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    self._memory_cache[query_hash] = cached_data
                    print(f"     💾 Cache HIT (disk)")
                    return cached_data
                except:
                    pass
        
        print(f"     🔍 Cache MISS: Fetching fresh data...")
        return None
    
    def _save_to_cache(self, query_hash: str, query: str, data: Dict[str, Any]):
        self._memory_cache[query_hash] = data
        
        if self.use_pgvector and self.vector_store:
            try:
                doc = Document(
                    page_content=json.dumps(data, ensure_ascii=False),
                    metadata={"query": query, "query_hash": query_hash, "timestamp": datetime.now().isoformat()}
                )
                self.vector_store.add_documents([doc])
                print(f"     💾 Cached to pgvector")
            except Exception as e:
                pass
        
        # Disk backup
        cache_file = self.cache_dir / f"{query_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        self.cache_index[query_hash] = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "file": f"{query_hash}.pkl"
        }
        self._save_cache_index()
    
    def _check_pgvector_for_relevant_docs(self, query: str, similarity_threshold: float = 0.7, k: int = 5) -> Dict[str, Any]:
        """
        Check pgvector database for relevant documents using semantic search
        
        Args:
            query: Search query
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            k: Number of top results to retrieve
            
        Returns:
            Dict with sources if found, None if no relevant docs
        """
        if not self.use_pgvector or not self.vector_store:
            return None
        
        try:
            print(f"     🔍 Searching pgvector RAG for relevant documents...")
            
            # Perform semantic similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not docs_with_scores:
                print(f"     ❌ No documents found in pgvector")
                return None
            
            # Filter by similarity threshold and extract relevant sources
            relevant_sources = []
            for doc, score in docs_with_scores:
                # Lower score means higher similarity in pgvector (distance metric)
                # Convert to similarity: similarity = 1 / (1 + distance)
                similarity = 1 / (1 + score)
                
                if similarity >= similarity_threshold:
                    try:
                        # Try to parse cached data
                        cached_data = json.loads(doc.page_content)
                        retrieved_data = cached_data.get('retrieved_data', {})
                        sources = retrieved_data.get('sources', [])
                        
                        # Add metadata about similarity
                        for source in sources:
                            source['similarity_score'] = round(similarity, 3)
                            source['from_cache'] = True
                        
                        relevant_sources.extend(sources)
                    except:
                        # If not cached data format, create source from document
                        relevant_sources.append({
                            "source_name": "Knowledge Base",
                            "content": doc.page_content[:500],
                            "credibility": "high",
                            "similarity_score": round(similarity, 3),
                            "from_cache": True
                        })
            
            if relevant_sources:
                print(f"     ✅ Found {len(relevant_sources)} relevant documents in pgvector (similarity >= {similarity_threshold})")
                
                # Create summary from relevant sources
                summaries = [s.get('content', '')[:200] for s in relevant_sources[:3]]
                summary = " ".join(summaries)
                
                # Extract key facts
                key_facts = []
                for source in relevant_sources[:3]:
                    content = source.get('content', '')
                    # Extract sentences as key facts
                    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                    key_facts.extend(sentences[:2])
                
                return {
                    "sources": relevant_sources,
                    "summary": summary[:500],
                    "key_facts": key_facts[:5],
                    "data_points": {},
                    "from_rag": True,
                    "rag_hits": len(relevant_sources)
                }
            else:
                print(f"     ⚠️  Documents found but similarity too low (< {similarity_threshold})")
                return None
                
        except Exception as e:
            print(f"     ⚠️  pgvector RAG search error: {e}")
            return None
    
    def _search_web_openai(self, query: str) -> Dict[str, Any]:
        try:
            print(f"     🌐 Performing OpenAI web search...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a web search agent. Return JSON:
                        {
                            "sources": [{"source_name": "...", "content": "...", "credibility": "high/medium/low"}],
                            "summary": "...",
                            "key_facts": ["fact1", "fact2"],
                            "data_points": {}
                        }"""
                    },
                    {"role": "user", "content": f"Search: {query}"}
                ]
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            return json.loads(content)
        except:
            return {"sources": [], "summary": "Search failed", "key_facts": []}
    
    def retrieve_information(self, query: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main retrieval method with semantic RAG search before web search
        
        Retrieval order:
        1. Check exact match in cache (by query hash)
        2. Check semantic similarity in pgvector RAG (similar documents)
        3. Perform web search (if no relevant docs found)
        4. Cache the results
        """
        query_hash = self._get_query_hash(query)
        
        # Step 1: Check exact cache match
        if not force_refresh:
            cached = self._get_from_cache(query_hash, query)
            if cached:
                return cached
        
        # Step 2: Check semantic similarity in pgvector RAG
        if not force_refresh:
            rag_results = self._check_pgvector_for_relevant_docs(query, similarity_threshold=0.7, k=5)
            if rag_results:
                print(f"     💾 Using {rag_results.get('rag_hits', 0)} semantically relevant documents from RAG")
                result = {
                    "agent": self.name,
                    "query": query,
                    "retrieved_data": rag_results,
                    "timestamp": datetime.now().isoformat(),
                    "cached": True,
                    "cache_type": "semantic_rag"
                }
                return result
        
        # Step 3: No cache or RAG matches - perform web search
        print(f"     🌐 No relevant docs in RAG - performing web search...")
        retrieved_data = self._search_web_openai(query)
        result = {
            "agent": self.name,
            "query": query,
            "retrieved_data": retrieved_data,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
        
        self._save_to_cache(query_hash, query, result)
        return result


# ============================================================================
# DEEP ANALYSIS TOOLS (USING @tool DECORATOR)
# ============================================================================

@tool
def comparative_analysis_tool(data: str) -> str:
    """
    Comparative Analysis Tool - Compares entities, concepts, or approaches.
    
    Args:
        data: JSON string containing data to analyze
        
    Returns:
        JSON string with comparison results including similarities, differences, 
        strengths/weaknesses, rankings, and recommendations
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    system_prompt = """You are a Comparative Analysis expert.
    
    Analyze and compare the information. Return JSON:
    {
        "comparison_matrix": {"entities": [], "dimensions": [], "scores": []},
        "similarities": [],
        "differences": [],
        "strengths_weaknesses": {},
        "ranking": [],
        "recommendation": "...",
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Compare:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"recommendation": response.content[:300], "confidence": 0.6})


@tool
def trend_analysis_tool(data: str) -> str:
    """
    Trend Analysis Tool - Identifies patterns and trends over time.
    
    Args:
        data: JSON string containing data to analyze
        
    Returns:
        JSON string with trends, patterns, projections, and momentum indicators
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    system_prompt = """You are a Trend Analysis expert.
    
    Identify patterns and trends. Return JSON:
    {
        "historical_trends": [],
        "current_state": "...",
        "emerging_patterns": [],
        "trend_direction": "upward/downward/stable",
        "momentum": {"speed": "...", "acceleration": "..."},
        "future_projection": {"short_term": "...", "medium_term": "...", "long_term": "..."},
        "key_drivers": [],
        "risks": [],
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze trends:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"current_state": response.content[:300], "confidence": 0.6})


@tool
def causal_reasoning_tool(data: str) -> str:
    """
    Causal Reasoning Tool - Determines cause-effect relationships.
    
    Args:
        data: JSON string containing data to analyze
        
    Returns:
        JSON string with causal chains, root causes, and contributing factors
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    system_prompt = """You are a Causal Reasoning expert.
    
    Identify cause-effect relationships. Return JSON:
    {
        "causal_chains": [],
        "root_causes": [],
        "contributing_factors": [],
        "mediating_variables": [],
        "confounding_factors": [],
        "causal_graph": {"nodes": [], "edges": []},
        "alternative_explanations": [],
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Determine causality:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"root_causes": [], "confidence": 0.6})


@tool
def statistical_analysis_tool(data: str) -> str:
    """
    Statistical Analysis Tool - Analyzes data patterns and distributions.
    
    Args:
        data: JSON string containing data to analyze
        
    Returns:
        JSON string with statistical measures, patterns, correlations, and outliers
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    system_prompt = """You are a Statistical Analysis expert.
    
    Analyze data patterns and statistics. Return JSON:
    {
        "descriptive_stats": {},
        "distribution": {"type": "...", "skewness": "...", "kurtosis": "..."},
        "patterns": [],
        "correlations": [],
        "outliers": [],
        "statistical_tests": {},
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Statistical analysis:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"descriptive_stats": {}, "confidence": 0.6})


# ============================================================================
# FACT CHECKING TOOLS (USING @tool DECORATOR)
# ============================================================================

@tool
def source_credibility_checker(data: str) -> str:
    """
    Source Credibility Checker - Evaluates source reliability.
    
    Args:
        data: JSON string containing sources to check
        
    Returns:
        JSON string with credibility ratings, authority indicators, and bias assessment
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    system_prompt = """You are a Source Credibility expert.
    
    Evaluate source credibility. Return JSON:
    {
        "overall_credibility": "high/medium/low",
        "source_ratings": [],
        "red_flags": [],
        "verified_sources": [],
        "questionable_sources": [],
        "recommendations": [],
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Check credibility:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"overall_credibility": "medium", "confidence": 0.6})


@tool
def cross_reference_validator(data: str) -> str:
    """
    Cross-Reference Validator - Validates claims across multiple sources.
    
    Args:
        data: JSON string containing claims to validate
        
    Returns:
        JSON string with validation status, agreement levels, and conflicts
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    system_prompt = """You are a Cross-Reference Validation expert.
    
    Validate claims across sources. Return JSON:
    {
        "cross_referenced_claims": [],
        "consensus_findings": [],
        "conflicts": [],
        "unverifiable_claims": [],
        "reliability_assessment": "high/medium/low",
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Cross-reference:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"reliability_assessment": "medium", "confidence": 0.6})


@tool
def confidence_score_calculator(data: str) -> str:
    """
    Confidence Score Calculator - Calculates weighted confidence scores.
    
    Args:
        data: JSON string containing data to assess
        
    Returns:
        JSON string with overall confidence, breakdown by factors, and uncertainty quantification
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    system_prompt = """You are a Confidence Score Calculator.
    
    Calculate confidence scores. Return JSON:
    {
        "overall_confidence": 0.8,
        "confidence_breakdown": {},
        "confidence_interval": {},
        "uncertainty_factors": [],
        "confidence_level": "high/medium/low",
        "reliability_grade": "A/B/C/D/F",
        "recommendations": []
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Calculate confidence:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"overall_confidence": 0.7, "reliability_grade": "B"})


@tool
def contradiction_detector(data: str) -> str:
    """
    Contradiction Detector - Identifies logical inconsistencies.
    
    Args:
        data: JSON string containing data to check
        
    Returns:
        JSON string with detected contradictions, severity, and resolution recommendations
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    system_prompt = """You are a Contradiction Detection expert.
    
    Detect contradictions and inconsistencies. Return JSON:
    {
        "contradictions_found": [],
        "logical_fallacies": [],
        "inconsistencies": [],
        "coherence_score": 0.8,
        "overall_assessment": "coherent/mostly coherent/some issues/major issues",
        "critical_issues": [],
        "requires_revision": false,
        "confidence": 0.8
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Detect contradictions:\n{data}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"coherence_score": 0.8, "requires_revision": False})


# ============================================================================
# OUTPUT FORMATTING TOOLS (USING @tool DECORATOR)
# ============================================================================

@tool
def report_structuring_tool(data: str) -> str:
    """
    Report Structuring Tool - Creates research paper structure.
    
    Args:
        data: JSON string containing research data
        
    Returns:
        JSON string with structured report sections (abstract, intro, findings, conclusions, etc.)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    system_prompt = """You are a Report Structuring expert.
    
    Create comprehensive research paper structure. Return JSON:
    {
        "title": "...",
        "abstract": {},
        "introduction": {},
        "methodology": {},
        "findings": {},
        "discussion": {},
        "conclusions": {},
        "recommendations": {}
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Structure report:\n{data[:2000]}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"title": "Research Report", "abstract": {}})


@tool
def citation_formatter(data: str) -> str:
    """
    Citation Formatter - Formats citations and bibliography.
    
    Args:
        data: JSON string containing sources
        
    Returns:
        JSON string with formatted citations in multiple styles (APA, MLA, Chicago)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    system_prompt = """You are a Citation Formatting expert.
    
    Format citations. Return JSON:
    {
        "sources_extracted": [],
        "in_text_citations": {},
        "references": {"apa": [], "mla": [], "chicago": []},
        "citation_count": 0,
        "unique_sources": 0
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Format citations:\n{data[:1000]}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"citation_count": 0, "references": {}})


@tool
def executive_summary_generator(data: str) -> str:
    """
    Executive Summary Generator - Creates multi-level summaries.
    
    Args:
        data: JSON string containing research findings
        
    Returns:
        JSON string with elevator pitch, one-page summary, extended summary, and key highlights
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    system_prompt = """You are an Executive Summary expert.
    
    Create multi-level summaries. Return JSON:
    {
        "elevator_pitch": {"text": "...", "word_count": 50},
        "one_page_summary": {},
        "extended_summary": {},
        "key_highlights": [],
        "strategic_insights": {},
        "talking_points": [],
        "quick_facts": []
    }"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate summaries:\n{data[:1500]}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return content
    except:
        return json.dumps({"elevator_pitch": {"text": "Summary", "word_count": 50}})


# ============================================================================
# LANGGRAPH NODE FUNCTIONS
# ============================================================================

# Initialize web scraper globally (shared across all nodes)
web_scraper = None

def init_web_scraper(api_key: str = None):
    """Initialize the web scraper"""
    global web_scraper
    if web_scraper is None:
        web_scraper = WebScraperAgent(api_key=api_key)


def research_coordinator_node(state: ResearchState) -> ResearchState:
    """
    Research Coordinator Agent: Orchestrates the entire research workflow
    Responsibilities:
    1. Initial query decomposition (if subtasks empty)
    2. Set current task for processing
    3. Check if more tasks remain and route accordingly
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Phase 1: Initial Decomposition (if subtasks not yet created)
    if not state.get('subtasks') or len(state['subtasks']) == 0:
        print("\n" + "="*80)
        print("🎯 RESEARCH COORDINATOR: ORCHESTRATING RESEARCH WORKFLOW")
        print("="*80)
        
        print(f"\n  🎯 ResearchCoordinator: Orchestrating research workflow")
        print(f"     Query: {state['original_query']}")
        print(f"     Max tasks: {state['page_limit']*2} (will decide optimal number)\n")
        
        system_prompt = """You are a Research Coordinator that decomposes research queries.
        
        Break down the query into 3-6 focused sub-tasks for comprehensive research. Return JSON array:
        [
            {"id": "task_1", "query": "specific sub-query", "priority": 1},
            {"id": "task_2", "query": "specific sub-query", "priority": 2}
        ]"""
        
        print("     🔍 Decomposing query into sub-tasks...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Decompose: {state['original_query']}")
        ])
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            subtasks = json.loads(content)
        except:
            subtasks = [{"id": "task_1", "query": state['original_query'], "priority": 1}]
        
        print(f"       ✓ Created {len(subtasks)} sub-tasks")
        print(f"     📊 Prioritizing {len(subtasks)} tasks...")
        print(f"       ✓ Prioritized with strategy: Sequential execution based on dependencies")
        print(f"     📈 Progress: 0.0% - Starting\n")
        
        return {
            **state,
            "subtasks": subtasks,
            "current_task_index": 0,
            "current_task": subtasks[0],
            "analysis_results": [],
            "validation_results": [],
            "next_step": "web_scraper"
        }
    
    # Phase 2: Task Management (set current task or check for more)
    current_index = state['current_task_index']
    subtasks = state['subtasks']
    
    # If current task just completed, check if more tasks remain
    if current_index < len(subtasks):
        # Set current task
        current_task = subtasks[current_index]
        
        return {
            **state,
            "current_task": current_task,
            "next_step": "web_scraper"
        }
    else:
        # All tasks completed, move to synthesis
        print(f"\n     📊 Final Progress: 100% - Synthesis Phase")
        return {
            **state,
            "next_step": "synthesize"
        }


def web_scraper_node(state: ResearchState) -> ResearchState:
    """
    Web Scraper: Retrieve information for current task
    """
    global web_scraper
    current_task = state['current_task']
    current_index = state['current_task_index']
    total_tasks = len(state['subtasks'])
    
    progress = (current_index / total_tasks) * 100
    
    print(f"\n     🎯 Delegating task: {current_task['id']}")
    print(f"        Query: {current_task['query']}")
    print(f"        📡 Step 1: Information Retrieval...")
    
    query = current_task['query']
    web_data = web_scraper.retrieve_information(query)
    
    num_sources = len(web_data.get('retrieved_data', {}).get('sources', []))
    print(f"     📚 Retrieved {num_sources} sources\n")
    
    return {
        **state,
        "web_data": web_data,
        "next_step": "deep_analysis"
    }


def deep_analysis_node(state: ResearchState) -> ResearchState:
    """
    Deep Analysis Agent: Analyze using all 4 analysis tools via ReAct agent
    Uses LLM to intelligently decide which tools to use and in what order
    """
    print(f"        🔬 Step 2: Deep Analysis...")
    print(f"     🤖 Using ReAct agent with 4 analysis tools...")
    
    current_task = state['current_task']
    web_data = state['web_data']
    
    # Prepare data for tools
    data_str = json.dumps(web_data.get('retrieved_data', {}), indent=2)[:1500]
    
    # Create LLM for the agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Define the analysis tools for the agent
    analysis_tools = [
        comparative_analysis_tool,
        trend_analysis_tool,
        causal_reasoning_tool,
        statistical_analysis_tool
    ]
    
    # Create system prompt for the agent
    system_prompt = SystemMessage(content="""You are a Deep Analysis expert. Your task is to perform comprehensive analysis using ALL 4 available tools.

CRITICAL INSTRUCTIONS:
1. You MUST use ALL 4 tools: comparative_analysis_tool, trend_analysis_tool, causal_reasoning_tool, and statistical_analysis_tool
2. Use each tool exactly once to analyze different aspects of the data
3. Pass the provided data to each tool
4. After using all 4 tools, provide a comprehensive synthesis

Your response should demonstrate use of all tools and provide integrated insights.""")

    # Create ReAct agent for deep analysis
    agent = create_react_agent(llm, analysis_tools)
    
    # Prepare agent input with explicit task to run ALL tools
    agent_input = {
        "messages": [
            SystemMessage(content="""You are a Deep Analysis expert. You have 4 analysis tools that you MUST use for every task.

MANDATORY WORKFLOW:
Step 1: Use comparative_analysis_tool - Compare different aspects and alternatives
Step 2: Use trend_analysis_tool - Identify patterns and trends over time
Step 3: Use causal_reasoning_tool - Analyze cause-effect relationships
Step 4: Use statistical_analysis_tool - Perform statistical analysis
Step 5: After using ALL 4 tools, synthesize the insights

YOU MUST COMPLETE ALL 5 STEPS. Do not skip any tool."""),
            HumanMessage(content=f"""Task: {current_task['query']}

Research Data to Analyze:
{data_str}

INSTRUCTIONS:
1. First, use comparative_analysis_tool with the above data
2. Then, use trend_analysis_tool with the above data
3. Then, use causal_reasoning_tool with the above data
4. Then, use statistical_analysis_tool with the above data
5. Finally, provide a comprehensive synthesis of all 4 analyses

Execute each step sequentially and use ALL 4 tools before providing your final answer.""")
        ]
    }
    
    # Run the agent
    try:
        agent_response = agent.invoke(agent_input)
        
        # Extract tool calls and results
        tools_used = []
        tools_results = {}
        
        for message in agent_response.get('messages', []):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', '')
                    if tool_name:
                        # Map tool names
                        clean_name = tool_name.replace('_tool', '')
                        tools_used.append(clean_name)
                        print(f"        🔄 Agent used: {clean_name}")
            
            # Collect tool responses
            if hasattr(message, 'content') and isinstance(message, ToolMessage):
                tool_name = getattr(message, 'name', 'unknown')
                clean_name = tool_name.replace('_tool', '')
                tools_results[clean_name] = message.content
                print(f"        ✅ {clean_name} complete")
        
        # Get final synthesis from agent
        final_message = agent_response['messages'][-1]
        synthesis = final_message.content if hasattr(final_message, 'content') else "Analysis completed"
        
        # If agent didn't use all tools, fall back to ensure all are called
        required_tools = ['comparative_analysis', 'trend_analysis', 'causal_reasoning', 'statistical_analysis']
        missing_tools = [t for t in required_tools if t not in tools_used]
        
        if missing_tools:
            print(f"     ⚠️  Agent missed tools: {missing_tools}. Running them now...")
            for tool_name in missing_tools:
                try:
                    if tool_name == 'comparative_analysis':
                        result = comparative_analysis_tool.invoke(data_str)
                    elif tool_name == 'trend_analysis':
                        result = trend_analysis_tool.invoke(data_str)
                    elif tool_name == 'causal_reasoning':
                        result = causal_reasoning_tool.invoke(data_str)
                    elif tool_name == 'statistical_analysis':
                        result = statistical_analysis_tool.invoke(data_str)
                    
                    tools_results[tool_name] = result
                    tools_used.append(tool_name)
                    print(f"        ✅ {tool_name} complete (fallback)")
                except Exception as e:
                    print(f"        ⚠️  {tool_name} error: {e}")
                    tools_results[tool_name] = json.dumps({"error": str(e)})
        
    except Exception as e:
        print(f"     ⚠️  Agent error: {e}. Falling back to direct tool calls...")
        # Fallback to direct calls if agent fails
        tools_results = {}
        tools_used = ['comparative_analysis', 'trend_analysis', 'causal_reasoning', 'statistical_analysis']
        
        for tool_name in tools_used:
            try:
                if tool_name == 'comparative_analysis':
                    result = comparative_analysis_tool.invoke(data_str)
                elif tool_name == 'trend_analysis':
                    result = trend_analysis_tool.invoke(data_str)
                elif tool_name == 'causal_reasoning':
                    result = causal_reasoning_tool.invoke(data_str)
                elif tool_name == 'statistical_analysis':
                    result = statistical_analysis_tool.invoke(data_str)
                
                tools_results[tool_name] = result
                print(f"        ✅ {tool_name} complete")
            except Exception as tool_error:
                print(f"        ⚠️  {tool_name} error: {tool_error}")
                tools_results[tool_name] = json.dumps({"error": str(tool_error)})
        
        synthesis = "Analysis completed via fallback mechanism"
    
    # Create final synthesis if not comprehensive enough
    print(f"     🧠 Final synthesis...")
    synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    synthesis_prompt = f"""Synthesize the following analysis results into comprehensive insights:

Task: {current_task['query']}

Comparative Analysis:
{tools_results.get('comparative_analysis', 'N/A')[:500]}

Trend Analysis:
{tools_results.get('trend_analysis', 'N/A')[:500]}

Causal Reasoning:
{tools_results.get('causal_reasoning', 'N/A')[:500]}

Statistical Analysis:
{tools_results.get('statistical_analysis', 'N/A')[:500]}

Agent Synthesis:
{synthesis[:500]}

Provide a comprehensive synthesis combining all analyses."""
    
    final_synthesis = synthesis_llm.invoke([
        SystemMessage(content="You are an expert at synthesizing multiple analysis results into coherent insights."),
        HumanMessage(content=synthesis_prompt)
    ])
    
    # Create analysis result
    analysis_result = {
        "task_id": current_task['id'],
        "query": current_task['query'],
        "web_data": web_data,
        "tools_used": tools_used,
        "tool_results": tools_results,
        "agent_synthesis": synthesis,
        "final_synthesis": final_synthesis.content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to analysis results
    analysis_results = state.get('analysis_results', [])
    analysis_results.append(analysis_result)
    
    print(f"        ✅ Analysis complete using: {', '.join(set(tools_used))}\n")
    
    return {
        **state,
        "analysis_results": analysis_results,
        "next_step": "fact_checking"
    }


def fact_checker_node(state: ResearchState) -> ResearchState:
    """
    Fact Checker Agent: Validate using all 4 validation tools via ReAct agent
    Uses LLM to intelligently decide how to use validation tools
    """
    print(f"        ✅ Step 3: Fact Checking...")
    print(f"     🤖 Using ReAct agent with 4 validation tools...")
    
    current_task = state['current_task']
    current_index = state['current_task_index']
    total_tasks = len(state['subtasks'])
    analysis_results = state['analysis_results']
    
    # Get latest analysis
    latest_analysis = analysis_results[-1] if analysis_results else {}
    
    # Prepare data for validation
    data_str = json.dumps(latest_analysis, indent=2, default=str)[:1500]
    
    # Create LLM for the agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Define the validation tools for the agent
    validation_tools = [
        source_credibility_checker,
        cross_reference_validator,
        confidence_score_calculator,
        contradiction_detector
    ]
    
    # Create ReAct agent for fact checking
    agent = create_react_agent(llm, validation_tools)
    
    # Prepare agent input with explicit task to run ALL tools
    agent_input = {
        "messages": [
            SystemMessage(content="""You are a Fact Checking expert. You have 4 validation tools that you MUST use for every task.

MANDATORY WORKFLOW:
Step 1: Use source_credibility_checker - Verify credibility of sources
Step 2: Use cross_reference_validator - Cross-check facts across sources
Step 3: Use confidence_score_calculator - Calculate confidence in findings
Step 4: Use contradiction_detector - Detect contradictions in data
Step 5: After using ALL 4 tools, synthesize the validation results

YOU MUST COMPLETE ALL 5 STEPS. Do not skip any tool."""),
            HumanMessage(content=f"""Task: Validate the analysis for {current_task['query']}

Analysis Data to Validate:
{data_str}

INSTRUCTIONS:
1. First, use source_credibility_checker with the above data
2. Then, use cross_reference_validator with the above data
3. Then, use confidence_score_calculator with the above data
4. Then, use contradiction_detector with the above data
5. Finally, provide a comprehensive validation assessment of all 4 checks

Execute each step sequentially and use ALL 4 tools before providing your final answer.""")
        ]
    }
    
    # Run the agent
    try:
        agent_response = agent.invoke(agent_input)
        
        # Extract tool calls and results
        tools_used = []
        tools_results = {}
        
        for message in agent_response.get('messages', []):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', '')
                    if tool_name:
                        # Map tool names
                        clean_name = tool_name.replace('_checker', '').replace('_validator', '').replace('_calculator', '').replace('_detector', '')
                        tools_used.append(clean_name)
                        print(f"        🔍 Agent used: {clean_name}")
            
            # Collect tool responses
            if hasattr(message, 'content') and isinstance(message, ToolMessage):
                tool_name = getattr(message, 'name', 'unknown')
                clean_name = tool_name.replace('_checker', '').replace('_validator', '').replace('_calculator', '').replace('_detector', '')
                tools_results[clean_name] = message.content
                print(f"        ✅ {clean_name} complete")
        
        # Get final synthesis from agent
        final_message = agent_response['messages'][-1]
        synthesis = final_message.content if hasattr(final_message, 'content') else "Validation completed"
        
        # If agent didn't use all tools, fall back to ensure all are called
        required_tools = ['source_credibility', 'cross_reference', 'confidence_score', 'contradiction']
        missing_tools = [t for t in required_tools if t not in [clean.replace('_check', '').replace('_score', '') for clean in tools_used]]
        
        if missing_tools:
            print(f"     ⚠️  Agent missed tools: {missing_tools}. Running them now...")
            for tool_name in missing_tools:
                try:
                    if 'source' in tool_name or 'credibility' in tool_name:
                        result = source_credibility_checker.invoke(data_str)
                        tools_results['source_credibility'] = result
                        tools_used.append('source_credibility_check')
                    elif 'cross' in tool_name or 'reference' in tool_name:
                        result = cross_reference_validator.invoke(data_str)
                        tools_results['cross_reference'] = result
                        tools_used.append('cross_reference')
                    elif 'confidence' in tool_name or 'score' in tool_name:
                        result = confidence_score_calculator.invoke(data_str)
                        tools_results['confidence_score'] = result
                        tools_used.append('confidence_score')
                    elif 'contradiction' in tool_name:
                        result = contradiction_detector.invoke(data_str)
                        tools_results['contradiction_detector'] = result
                        tools_used.append('contradiction_detector')
                    
                    print(f"        ✅ {tool_name} complete (fallback)")
                except Exception as e:
                    print(f"        ⚠️  {tool_name} error: {e}")
                    tools_results[tool_name] = json.dumps({"error": str(e)})
        
    except Exception as e:
        print(f"     ⚠️  Agent error: {e}. Falling back to direct tool calls...")
        # Fallback to direct calls if agent fails
        tools_results = {}
        tools_used = ['source_credibility_check', 'cross_reference', 'confidence_score', 'contradiction_detector']
        
        for tool_name in tools_used:
            try:
                if tool_name == 'source_credibility_check':
                    result = source_credibility_checker.invoke(data_str)
                    tools_results['source_credibility'] = result
                elif tool_name == 'cross_reference':
                    result = cross_reference_validator.invoke(data_str)
                    tools_results['cross_reference'] = result
                elif tool_name == 'confidence_score':
                    result = confidence_score_calculator.invoke(data_str)
                    tools_results['confidence_score'] = result
                elif tool_name == 'contradiction_detector':
                    result = contradiction_detector.invoke(data_str)
                    tools_results['contradiction_detector'] = result
                
                print(f"        ✅ {tool_name} complete")
            except Exception as tool_error:
                print(f"        ⚠️  {tool_name} error: {tool_error}")
                tools_results[tool_name] = json.dumps({"error": str(tool_error)})
        
        synthesis = "Validation completed via fallback mechanism"
    
    # Create final synthesis if not comprehensive enough
    print(f"     🧠 Final validation synthesis...")
    synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    synthesis_prompt = f"""Synthesize the following validation results:

Task: {current_task['query']}

Source Credibility:
{tools_results.get('source_credibility', 'N/A')[:300]}

Cross Reference:
{tools_results.get('cross_reference', 'N/A')[:300]}

Confidence Score:
{tools_results.get('confidence_score', 'N/A')[:300]}

Contradiction Detection:
{tools_results.get('contradiction_detector', 'N/A')[:300]}

Agent Synthesis:
{synthesis[:500]}

Provide overall validation assessment combining all checks."""
    
    final_synthesis = synthesis_llm.invoke([
        SystemMessage(content="You are an expert at validating research findings."),
        HumanMessage(content=synthesis_prompt)
    ])
    
    # Create validation result
    validation_result = {
        "task_id": current_task['id'],
        "analysis": latest_analysis,
        "tools_used": tools_used,
        "tool_results": tools_results,
        "agent_synthesis": synthesis,
        "final_synthesis": final_synthesis.content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to validation results
    validation_results = state.get('validation_results', [])
    validation_results.append(validation_result)
    
    # Calculate progress
    progress = ((current_index + 1) / total_tasks) * 100
    
    # Determine phase
    if progress <= 25:
        phase = "Early Research"
    elif progress <= 50:
        phase = "Deep Analysis"
    elif progress <= 75:
        phase = "Validation & Synthesis"
    else:
        phase = "Finalizing"
    
    print(f"        ✅ Validation complete using: {', '.join(tools_used)}")
    print(f"     📈 Progress: {progress:.2f}% - {phase}\n")
    
    return {
        **state,
        "validation_results": validation_results,
        "next_step": "task_complete"
    }


def task_completion_handler(state: ResearchState) -> ResearchState:
    """
    Handles task completion and moves to next task
    Updates the task index after a task is validated
    """
    current_index = state['current_task_index']
    next_index = current_index + 1
    
    return {
        **state,
        "current_task_index": next_index,
        "next_step": "research_coordinator"
    }


def synthesize_results(state: ResearchState) -> ResearchState:
    """
    Manager Agent: Synthesize all results
    """
    print(f"     � Synthesizing results from {len(state['subtasks'])} tasks...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    analysis_results = state.get('analysis_results', [])
    validation_results = state.get('validation_results', [])
    
    system_prompt = """You are a Manager Agent synthesizing research findings.
    
    Combine insights from all tasks. Return JSON:
    {
        "overall_summary": "...",
        "combined_insights": [],
        "key_findings": [],
        "cross_task_patterns": [],
        "synthesized_recommendations": [],
        "confidence_score": 0.8
    }"""
    
    # Prepare synthesis data
    synthesis_data = {
        "total_tasks": len(analysis_results),
        "analyses": [{"task_id": a['task_id'], "query": a['query']} for a in analysis_results],
        "validations": [{"task_id": v['task_id']} for v in validation_results]
    }
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Synthesize:\n{json.dumps(synthesis_data, indent=2)}\n\nAnalyses: {json.dumps(analysis_results, indent=2, default=str)[:2000]}")
    ])
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        synthesized = json.loads(content)
    except:
        synthesized = {
            "overall_summary": response.content[:500],
            "combined_insights": [],
            "key_findings": [],
            "confidence_score": 0.7
        }
    
    confidence_level = "High" if synthesized.get('confidence_score', 0.7) >= 0.8 else "Medium" if synthesized.get('confidence_score', 0.7) >= 0.6 else "Low"
    
    print(f"       ✓ Synthesis complete - Confidence: {confidence_level}")
    print(f"     📈 Progress: 100.0% - Complete\n")
    
    print(f"✅ Manager orchestration complete!")
    print(f"   • Tasks: {len(state['subtasks'])}")
    print(f"   • Confidence: {confidence_level}\n")
    
    return {
        **state,
        "synthesized_result": synthesized,
        "next_step": "format_output"
    }


def output_formatter_node(state: ResearchState) -> ResearchState:
    """
    Output Formatter Agent: Format using ReAct agent with tools
    """
    print("="*80)
    print("📝 FORMATTING FINAL OUTPUT")
    print("="*80)
    print("  📝 OutputFormattingAgent: Formatting output...")
    
    # Tools for formatting (excluding visualization for concise output)
    tools = [
        report_structuring_tool,
        citation_formatter,
        executive_summary_generator
    ]
    
    # Create ReAct agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    agent = create_react_agent(llm, tools)
    
    synthesized = state.get('synthesized_result', {})
    page_limit = state.get('page_limit', 3)
    
    # Prepare input for agent
    input_data = {
        "messages": [
            HumanMessage(content=f"""Format this research into a {page_limit}-page report:

Synthesized Results:
{json.dumps(synthesized, indent=2)[:1500]}

Use your formatting tools to:
1. Structure the report with proper sections
2. Format citations
3. Generate executive summary

Create a comprehensive research report.""")
        ]
    }
    
    # Run agent
    print(f"        📄 Running formatting tools...")
    result = agent.invoke(input_data)
    
    # Extract formatted output
    final_output = {
        "document_type": f"{page_limit}-Page Research Report",
        "metadata": {
            "generated_date": datetime.now().strftime("%B %d, %Y"),
            "page_limit": page_limit,
            "format": "Text-only"
        },
        "agent_response": result,
        "synthesized_data": synthesized,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"     ✅ Formatting complete\n")
    print(f"✅ Final output formatted\n")
    
    return {
        **state,
        "final_output": final_output,
        "next_step": "end"
    }


def route_next_step(state: ResearchState) -> str:
    """Router function to determine next node"""
    next_step = state.get('next_step', 'end')
    
    routing = {
        "web_scraper": "web_scraper",
        "deep_analysis": "deep_analysis",
        "fact_checking": "fact_checker",
        "task_complete": "task_complete",
        "research_coordinator": "research_coordinator",
        "synthesize": "synthesize",
        "format_output": "output_formatter",
        "end": END
    }
    
    return routing.get(next_step, END)


# ============================================================================
# BUILD LANGGRAPH
# ============================================================================

def create_research_graph(api_key: str = None) -> StateGraph:
    """
    Create the LangGraph workflow with Research Coordinator
    """
    # Initialize web scraper
    init_web_scraper(api_key)
    
    # Create graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("research_coordinator", research_coordinator_node)
    graph.add_node("web_scraper", web_scraper_node)
    graph.add_node("deep_analysis", deep_analysis_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("task_complete", task_completion_handler)
    graph.add_node("synthesize", synthesize_results)
    graph.add_node("output_formatter", output_formatter_node)
    
    # Add edges
    graph.add_edge(START, "research_coordinator")
    graph.add_edge("web_scraper", "deep_analysis")
    graph.add_edge("deep_analysis", "fact_checker")
    graph.add_edge("fact_checker", "task_complete")
    graph.add_edge("task_complete", "research_coordinator")
    
    # Conditional routing from research_coordinator
    graph.add_conditional_edges(
        "research_coordinator",
        route_next_step,
        {
            "web_scraper": "web_scraper",
            "synthesize": "synthesize",
            END: END
        }
    )
    
    graph.add_edge("synthesize", "output_formatter")
    graph.add_edge("output_formatter", END)
    
    # Compile
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_research(query: str, api_key: str = None, page_limit: int = 3, 
                include_visualizations: bool = False) -> Dict[str, Any]:
    """
    Run complete research workflow using LangGraph
    
    Args:
        query: Research question
        api_key: OpenAI API key
        page_limit: Number of pages (1-20)
        include_visualizations: Whether to include visualizations
        
    Returns:
        Complete research results
    """
    print("\n" + "="*80)
    print("🤖 MULTI-AGENT RESEARCH SYSTEM (LangGraph)")
    print("="*80)
    print(f"\n📝 Query: {query}")
    print(f"📄 Page Limit: {page_limit}")
    print(f"📊 Visualizations: {'Enabled' if include_visualizations else 'Disabled'}\n")
    
    # Create graph
    graph = create_research_graph(api_key)
    
    # Initial state
    initial_state = {
        "original_query": query,
        "page_limit": page_limit,
        "include_visualizations": include_visualizations,
        "subtasks": [],
        "current_task_index": 0,
        "current_task": {},
        "web_data": {},
        "analysis_results": [],
        "validation_results": [],
        "synthesized_result": {},
        "final_output": {},
        "messages": [],
        "next_step": ""
    }
    
    # Run graph with increased recursion limit
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 100  # Increase from default 25 to handle multiple tasks
    }
    final_state = graph.invoke(initial_state, config)
    
    return {
        "original_query": query,
        "page_limit": page_limit,
        "include_visualizations": include_visualizations,
        "subtasks": final_state.get('subtasks', []),
        "tasks_processed": len(final_state.get('analysis_results', [])),
        "analysis_results": final_state.get('analysis_results', []),
        "validation_results": final_state.get('validation_results', []),
        "synthesized_result": final_state.get('synthesized_result', {}),
        "final_output": final_state.get('final_output', {}),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Run research
    results = run_research(
        query="What is the impact of AI on education?",
        api_key=api_key,
        page_limit=3,
        include_visualizations=False
    )
    
    # Save results
    output_file = "research_langgraph_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

"""
LangGraph Integration with Multi-Agent System
This shows how to integrate the Multi-Agent Research System into a LangGraph workflow

System Architecture:
- Manager Agent (orchestrates the workflow)
- Deep Analysis Agent (performs analysis, uses Web Scraper)
- Web Scraper Agent (retrieves information)
- Fact Checking Agent (validates results)
- Output Formatting Agent (formats final output)

Workflow:
1. Manager decomposes query into sub-tasks
2. For each sub-task:
   a. Deep Analysis Agent analyzes (uses Web Scraper internally)
   b. Fact Checking Agent validates the analysis
3. Manager synthesizes all results
4. Output Formatting Agent formats the final output
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import dotenv
import operator
from multi_agent_system import (
    ManagerAgent,
    DeepAnalysisAgent,
    FactCheckingAgent,
    OutputFormattingAgent,
    WebScraperAgent,
    SubTask
)
import os
import json



dotenv.load_dotenv()    

# ============================================================================
# STATE DEFINITION
# ============================================================================

class MultiAgentState(TypedDict):
    """State for the multi-agent LangGraph workflow"""
    query: str
    sub_tasks: List[SubTask]
    current_task_index: int
    completed_tasks: List[SubTask]
    synthesized_results: Dict[str, Any]
    final_output: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Agents
    manager_agent: ManagerAgent
    deep_analysis_agent: DeepAnalysisAgent
    fact_checking_agent: FactCheckingAgent
    output_formatting_agent: OutputFormattingAgent
    web_scraper_agent: WebScraperAgent


# ============================================================================
# GRAPH NODES
# ============================================================================

def initialize_agents_node(state: MultiAgentState) -> MultiAgentState:
    """Initialize all agents in the system"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("\n" + "="*80)
    print("🚀 INITIALIZING MULTI-AGENT SYSTEM")
    print("="*80)
    
    # Initialize Manager Agent (which initializes all sub-agents)
    manager = ManagerAgent(api_key=api_key)
    
    state["manager_agent"] = manager
    state["deep_analysis_agent"] = manager.deep_analysis_agent
    state["fact_checking_agent"] = manager.fact_checking_agent
    state["output_formatting_agent"] = manager.output_formatting_agent
    state["web_scraper_agent"] = manager.deep_analysis_agent.web_scraper
    
    state["current_task_index"] = 0
    state["completed_tasks"] = []
    
    print("✅ Agents initialized:")
    print("   • Manager Agent (Orchestrator)")
    print("   • Deep Analysis Agent")
    print("   • Web Scraper & Document Retrieval Agent")
    print("   • Fact Checking & Validation Agent")
    print("   • Output Formatting Agent")
    
    return state


def decompose_query_node(state: MultiAgentState) -> MultiAgentState:
    """Manager agent decomposes the query into sub-tasks"""
    manager: ManagerAgent = state["manager_agent"]
    
    print("\n" + "─"*80)
    print("📋 STEP 1: QUERY DECOMPOSITION")
    print("─"*80)
    print(f"Query: {state['query']}\n")
    
    sub_tasks = manager.decompose_query(state["query"])
    state["sub_tasks"] = sub_tasks
    
    print(f"\n✅ Created {len(sub_tasks)} sub-tasks:")
    for i, task in enumerate(sub_tasks, 1):
        print(f"   {i}. [{task.id}] {task.query} (Priority: {task.priority})")
    
    return state


def process_task_node(state: MultiAgentState) -> MultiAgentState:
    """
    Process a single task through the agent pipeline:
    1. Deep Analysis Agent (which uses Web Scraper Agent internally)
    2. Fact Checking Agent validates the analysis
    """
    tasks = state["sub_tasks"]
    current_index = state["current_task_index"]
    
    if current_index < len(tasks):
        task = tasks[current_index]
        
        print("\n" + "─"*80)
        print(f"🔄 STEP 2.{current_index + 1}: PROCESSING TASK {current_index + 1}/{len(tasks)}")
        print("─"*80)
        print(f"Task ID: {task.id}")
        print(f"Query: {task.query}\n")
        
        # Get agents
        deep_analysis_agent: DeepAnalysisAgent = state["deep_analysis_agent"]
        fact_checking_agent: FactCheckingAgent = state["fact_checking_agent"]
        
        # Step 2a: Deep Analysis (includes web scraping internally)
        print("┌─ Deep Analysis Phase ─────────────────────────────────────┐")
        task.status = "analyzing"
        analysis_result = deep_analysis_agent.analyze_task(task)
        task.deep_analysis_result = analysis_result
        print("└────────────────────────────────────────────────────────────┘")
        
        # Step 2b: Fact Checking
        print("\n┌─ Fact Checking Phase ──────────────────────────────────────┐")
        task.status = "fact_checking"
        fact_check_result = fact_checking_agent.validate_analysis(analysis_result)
        task.fact_check_result = fact_check_result
        print("└────────────────────────────────────────────────────────────┘")
        
        # Mark as completed
        task.status = "completed"
        
        # Update state
        state["sub_tasks"][current_index] = task
        state["completed_tasks"].append(task)
        state["current_task_index"] = current_index + 1
        
        # Display validation summary
        validation = fact_check_result.get('validation', {})
        print(f"\n📊 Task Summary:")
        print(f"   Status: {validation.get('validation_status', 'unknown')}")
        print(f"   Accuracy: {validation.get('accuracy_score', 0):.2f}")
        print(f"   Recommendation: {validation.get('recommendation', 'unknown')}")
    
    return state


def should_continue_processing(state: MultiAgentState) -> str:
    """Conditional edge to determine if more tasks need processing"""
    if state["current_task_index"] < len(state["sub_tasks"]):
        return "continue"
    else:
        return "synthesize"


def synthesize_results_node(state: MultiAgentState) -> MultiAgentState:
    """Manager agent synthesizes results from all completed tasks"""
    manager: ManagerAgent = state["manager_agent"]
    completed_tasks = state["completed_tasks"]
    
    print("\n" + "─"*80)
    print("🔮 STEP 3: RESULT SYNTHESIS")
    print("─"*80)
    print(f"Synthesizing results from {len(completed_tasks)} completed tasks...\n")
    
    synthesized_results = manager.synthesize_results(completed_tasks)
    state["synthesized_results"] = synthesized_results
    
    print("✅ Synthesis complete")
    print(f"   Overall confidence: {synthesized_results.get('confidence_score', 0):.2f}")
    print(f"   Key findings: {len(synthesized_results.get('key_findings', []))}")
    print(f"   Recommendations: {len(synthesized_results.get('synthesized_recommendations', []))}")
    
    return state


def format_output_node(state: MultiAgentState) -> MultiAgentState:
    """Output Formatting Agent formats the final results"""
    output_agent: OutputFormattingAgent = state["output_formatting_agent"]
    synthesized_results = state["synthesized_results"]
    
    print("\n" + "─"*80)
    print("📝 STEP 4: OUTPUT FORMATTING")
    print("─"*80)
    
    final_output = output_agent.format_output(synthesized_results)
    state["final_output"] = final_output
    
    print("✅ Output formatting complete")
    
    return state


def display_final_results_node(state: MultiAgentState) -> MultiAgentState:
    """Display the final formatted results"""
    final_output = state["final_output"]
    formatted = final_output.get("formatted_output", {})
    
    print("\n" + "="*80)
    print("📊 FINAL RESEARCH RESULTS")
    print("="*80)
    
    # Executive Summary
    print("\n📌 EXECUTIVE SUMMARY")
    print("─"*80)
    exec_summary = formatted.get("executive_summary", "N/A")
    print(f"{exec_summary}\n")
    
    # Key Takeaways
    print("🎯 KEY TAKEAWAYS")
    print("─"*80)
    for i, takeaway in enumerate(formatted.get("key_takeaways", []), 1):
        print(f"{i}. {takeaway}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("─"*80)
    for i, rec in enumerate(formatted.get("recommendations", []), 1):
        if isinstance(rec, dict):
            priority = rec.get('priority', 'medium').upper()
            text = rec.get('recommendation', str(rec))
            print(f"{i}. [{priority}] {text}")
        else:
            print(f"{i}. {rec}")
    
    # Confidence Metrics
    print("\n📈 CONFIDENCE METRICS")
    print("─"*80)
    metrics = formatted.get("confidence_metrics", {})
    print(f"Overall Confidence: {metrics.get('overall_confidence', 0):.2f}")
    print(f"Data Quality: {metrics.get('data_quality', 0):.2f}")
    print(f"Validation Score: {metrics.get('validation_score', 0):.2f}")
    
    # Summary Stats
    print("\n📋 RESEARCH SUMMARY")
    print("─"*80)
    appendix = formatted.get("appendix", {})
    print(f"Tasks Analyzed: {appendix.get('tasks_analyzed', len(state['completed_tasks']))}")
    print(f"Sources Retrieved: {appendix.get('sources_count', 'N/A')}")
    print(f"Total Insights: {appendix.get('total_insights', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ MULTI-AGENT RESEARCH WORKFLOW COMPLETE")
    print("="*80 + "\n")
    
    return state


# ============================================================================
# WORKFLOW CONSTRUCTION
# ============================================================================

def create_multi_agent_workflow() -> StateGraph:
    """Create and configure the multi-agent LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_agents_node)
    workflow.add_node("decompose", decompose_query_node)
    workflow.add_node("process_task", process_task_node)
    workflow.add_node("synthesize", synthesize_results_node)
    workflow.add_node("format_output", format_output_node)
    workflow.add_node("display_results", display_final_results_node)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "decompose")
    workflow.add_edge("decompose", "process_task")
    
    # Conditional edge: continue processing tasks or move to synthesis
    workflow.add_conditional_edges(
        "process_task",
        should_continue_processing,
        {
            "continue": "process_task",      # Loop back to process next task
            "synthesize": "synthesize"       # Move to synthesis
        }
    )
    
    workflow.add_edge("synthesize", "format_output")
    workflow.add_edge("format_output", "display_results")
    workflow.add_edge("display_results", END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_multi_agent_research(query: str) -> Dict[str, Any]:
    """
    Run the complete multi-agent research workflow
    
    Args:
        query: The research query to process
        
    Returns:
        Final research results
    """
    # Create the workflow
    app = create_multi_agent_workflow()
    
    # Initialize state
    initial_state = MultiAgentState(
        query=query,
        sub_tasks=[],
        current_task_index=0,
        completed_tasks=[],
        synthesized_results={},
        final_output={},
        messages=[],
        manager_agent=None,
        deep_analysis_agent=None,
        fact_checking_agent=None,
        output_formatting_agent=None,
        web_scraper_agent=None
    )
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    return {
        "query": final_state["query"],
        "tasks_processed": len(final_state["completed_tasks"]),
        "final_output": final_state["final_output"],
        "synthesis": final_state["synthesized_results"]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Error: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it before running:")
        print('$env:OPENAI_API_KEY="your-api-key-here"')
        exit(1)
    
    # Example research query
    research_query = """
        COVID-19 economic impact on developing countries
    """
    
    print("🔬 Multi-Agent Research System - LangGraph Integration")
    print("─"*80)
    
    # Run the workflow
    result = run_multi_agent_research(research_query.strip())
    
    # Save results to file
    output_file = "research_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("\n✨ Workflow completed successfully!")

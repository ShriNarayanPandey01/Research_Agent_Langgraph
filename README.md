# Research Orchestration Agent for LangGraph

A powerful AI agent designed to manage and orchestrate research workflows using LangChain and LangGraph. This agent breaks down complex queries into manageable sub-tasks, prioritizes them, tracks progress, and synthesizes results into comprehensive findings.

## 🎯 Features

- **Query Decomposition**: Automatically breaks down complex research queries into smaller, manageable sub-queries
- **Task Prioritization**: Intelligently prioritizes tasks based on dependencies, complexity, and importance
- **Progress Tracking**: Real-time tracking of task execution and progress
- **Result Synthesis**: Combines findings from multiple tasks into coherent, actionable insights
- **LangGraph Integration**: Designed as an entry point for LangGraph workflows
- **Modular Design**: Each tool can be used independently or as part of the complete workflow

## 📋 Components

### Core Tools

1. **Query Decomposition Tool** (`query_decomposition_tool`)
   - Breaks complex queries into sub-queries
   - Identifies dependencies between tasks
   - Creates actionable, focused research items

2. **Task Prioritization Tool** (`task_prioritization_tool`)
   - Assigns priority levels (1-5) to each task
   - Considers dependencies and complexity
   - Provides reasoning for prioritization

3. **Progress Tracking Tool** (`progress_tracking_tool`)
   - Tracks status (pending, in_progress, completed, failed)
   - Records progress percentage and notes
   - Timestamps all updates

4. **Result Synthesis Tool** (`result_synthesis_tool`)
   - Synthesizes findings into executive summaries
   - Extracts key findings and patterns
   - Generates actionable recommendations
   - Provides confidence scores

### Main Agent Class

`ResearchOrchestrationAgent` - The main orchestration class that:
- Manages workflow state
- Coordinates all tools
- Provides high-level methods for each workflow step
- Integrates seamlessly with LangGraph

## 🚀 Installation

1. **Clone or download the files**

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**:
   ```powershell
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   ```

## 📖 Usage

### Basic Usage

```python
from research_agent import create_research_agent
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create agent
agent = create_research_agent()

# Step 1: Decompose query
query = "What are the best practices for microservices architecture?"
sub_queries = agent.decompose_query(query)

# Step 2: Prioritize tasks
tasks = agent.prioritize_tasks()

# Step 3: Execute and track (you implement the actual research)
for task in tasks:
    # Your research logic here
    result = "Your research findings..."
    agent.add_task_result(task['id'], result)

# Step 4: Synthesize results
final_results = agent.synthesize_results()
print(final_results)
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph, END
from research_agent import ResearchOrchestrationAgent

# Define your graph nodes
def decompose_node(state):
    agent = state["agent"]
    sub_queries = agent.decompose_query(state["query"])
    state["sub_queries"] = sub_queries
    return state

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_node)
# ... add more nodes

# See langgraph_integration.py for complete example
```

### Running Examples

**Example 1: Basic workflow**
```powershell
python example_usage.py
```

**Example 2: LangGraph integration**
```powershell
python langgraph_integration.py
```

## 📁 File Structure

```
├── research_agent.py           # Main agent implementation
├── example_usage.py            # Basic usage examples
├── langgraph_integration.py    # LangGraph workflow example
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 API Reference

### ResearchOrchestrationAgent

#### Methods

**`__init__(openai_api_key: str = None)`**
- Initialize the agent with optional API key

**`decompose_query(query: str) -> List[Dict[str, Any]]`**
- Decomposes a query into sub-queries
- Returns list of sub-queries with IDs, descriptions, and dependencies

**`prioritize_tasks() -> List[Dict[str, Any]]`**
- Prioritizes the decomposed tasks
- Returns tasks with priority levels and reasoning

**`update_progress(task_id: str, status: str, progress: int, notes: str = "") -> Dict`**
- Updates progress for a specific task
- Status: "pending", "in_progress", "completed", "failed"
- Progress: 0-100

**`add_task_result(task_id: str, result: str)`**
- Adds research findings for a completed task

**`synthesize_results() -> Dict[str, Any]`**
- Synthesizes all results into final output
- Returns summary, key findings, and recommendations

**`get_state() -> ResearchAgentState`**
- Returns current agent state

**`get_overall_progress() -> Dict[str, Any]`**
- Returns overall progress metrics

**`reset()`**
- Resets agent to initial state

## 🎨 Data Models

### SubQuery
```python
{
    "id": "sq1",
    "query": "What are microservices?",
    "description": "Define and explain microservices",
    "dependencies": []
}
```

### PrioritizedTask
```python
{
    "id": "sq1",
    "query": "What are microservices?",
    "priority": 1,
    "reasoning": "Foundation for other tasks",
    "estimated_complexity": "low"
}
```

### TaskProgress
```python
{
    "id": "sq1",
    "status": "completed",
    "progress_percentage": 100,
    "notes": "Research findings here...",
    "updated_at": "2025-10-24T10:30:00"
}
```

### SynthesizedResult
```python
{
    "summary": "Executive summary...",
    "key_findings": ["finding 1", "finding 2"],
    "detailed_results": {"sq1": "details..."},
    "recommendations": ["recommendation 1"],
    "confidence_score": 0.85
}
```

## 🔄 Workflow Example

```
1. Initialize Agent
        ↓
2. Decompose Query
   → Sub-query 1
   → Sub-query 2
   → Sub-query 3
        ↓
3. Prioritize Tasks
   → Priority 1: Sub-query 2
   → Priority 2: Sub-query 1
   → Priority 3: Sub-query 3
        ↓
4. Execute & Track
   → Execute Priority 1 (0% → 100%)
   → Execute Priority 2 (0% → 100%)
   → Execute Priority 3 (0% → 100%)
        ↓
5. Synthesize Results
   → Generate summary
   → Extract key findings
   → Create recommendations
        ↓
6. Return Final Output
```

## 🛠️ Integration with LangGraph

The agent is designed to work seamlessly with LangGraph:

1. **Entry Point**: Use `research_agent_entry_point()` function
2. **State Management**: Agent maintains its own state compatible with LangGraph
3. **Node Functions**: Each method can be a LangGraph node
4. **Conditional Edges**: Use task status for routing
5. **Loops**: Easily implement task execution loops

See `langgraph_integration.py` for a complete working example.

## 🎯 Use Cases

- **Research Automation**: Automate complex research tasks
- **Multi-Agent Systems**: Use as coordinator in multi-agent setups
- **Knowledge Synthesis**: Combine information from multiple sources
- **Task Orchestration**: Manage complex workflows with dependencies
- **LangGraph Workflows**: Entry point for sophisticated AI workflows

## 📝 Notes

- The agent uses GPT-4o-mini by default for cost efficiency
- You can modify the model in the code if needed
- All tools can be used independently
- State is maintained throughout the workflow
- Progress tracking enables monitoring and debugging

## 🤝 Contributing

Feel free to extend the agent with:
- Custom tools for specific research needs
- Different LLM providers
- Enhanced progress tracking
- Custom synthesis strategies

## 📄 License

This code is provided as-is for use in your projects.

## 🆘 Troubleshooting

**Issue**: Import errors for langchain packages
**Solution**: Run `pip install -r requirements.txt`

**Issue**: OpenAI API errors
**Solution**: Verify your API key is set correctly and has credits

**Issue**: JSON parsing errors in tool outputs
**Solution**: The tools include fallback parsing for markdown-wrapped JSON

## 📧 Support

For questions or issues, refer to:
- LangChain documentation: https://python.langchain.com/
- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- OpenAI API documentation: https://platform.openai.com/docs

---

**Happy Researching! 🚀**

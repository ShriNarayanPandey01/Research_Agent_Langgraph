# 🤖 Multi-Agent Research System with LangGraph

A sophisticated multi-agent AI system that orchestrates research workflows using specialized agents. Built with LangChain and LangGraph for production-ready AI applications.

## ✨ New Features (v2.0)

### 📄 Flexible Output Formats
- **Concise Reports** (1-3 pages) - Default format for quick insights
- **Medium Reports** (4-7 pages) - Balanced detail and brevity
- **Research Papers** (8-20 pages) - Comprehensive academic format

### 🎨 Customization Options
- **Page Limits**: Specify exact page count (1-20 pages)
- **Visualization Control**: Enable/disable charts and graphs
- **Text-Only Reports**: Faster generation without visualizations

### ⚡ Quick Start
```python
from multi_agent_system import run_research

# Default: 3-page concise report (no visualizations)
results = run_research(query="Your research question", api_key="your-key")

# Custom: 5-page medium report
results = run_research(query="Your question", api_key="your-key", page_limit=5)

# Full: 15-page research paper with visualizations
results = run_research(
    query="Your question",
    api_key="your-key",
    page_limit=15,
    include_visualizations=True
)
```

📖 **See [CONCISE_OUTPUT_GUIDE.md](CONCISE_OUTPUT_GUIDE.md) for detailed usage examples**

---

## 🎯 System Overview

This system implements a **Manager-Worker architecture** with 5 specialized AI agents:

### **Manager Agent** (Orchestrator)
- Decomposes complex queries into sub-tasks
- Coordinates all sub-agents
- Synthesizes final results
- Ensures workflow completion

### **Sub-Agents**

1. **🌐 Web Scraper & Document Retrieval Agent**
   - Retrieves information from multiple sources
   - Simulates web scraping and document search
   - Provides credibility indicators
   - Returns structured data

2. **🔬 Deep Analysis Agent**
   - Performs in-depth analysis on retrieved data
   - Uses Web Scraper internally for information gathering
   - Generates insights and conclusions
   - Provides confidence levels

3. **✓ Fact Checking & Validation Agent**
   - Validates analysis results
   - Checks logical consistency
   - Flags unsupported claims
   - Recommends approval/revision

4. **📝 Output Formatting Agent**
   - Creates professional reports
   - Formats executive summaries
   - Organizes findings hierarchically
   - Adds confidence metrics

---

## 🔄 Workflow Architecture

```
┌─────────────────────────────────────────────┐
│         USER QUERY                          │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  MANAGER AGENT: Decompose Query            │
│  → Creates sub-tasks with priorities       │
└──────────────────┬──────────────────────────┘
                   ▼
         ┌─────────────────────┐
         │  FOR EACH SUB-TASK  │
         └─────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  DEEP ANALYSIS AGENT                        │
│  ├─ Calls Web Scraper Agent internally     │
│  ├─ Analyzes retrieved information          │
│  └─ Generates insights & conclusions        │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  FACT CHECKING AGENT                        │
│  ├─ Validates analysis results              │
│  ├─ Checks consistency                      │
│  └─ Flags errors or approves                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         (Loop back for next task)
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  MANAGER AGENT: Synthesize Results          │
│  → Combines all validated task results     │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  OUTPUT FORMATTING AGENT                    │
│  → Creates professional final report        │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│         FINAL OUTPUT TO USER                │
└─────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- OpenAI API key

### Install Dependencies
```powershell
pip install langchain langchain-core langchain-openai langgraph openai pydantic python-dotenv
```

### Set API Key
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

---

## 🚀 Quick Start

### Option 1: Simple Usage (Recommended)
```python
from multi_agent_system import run_research

# One-line execution - Manager handles everything
result = run_research("What are best practices for microservices architecture?")

print(result['final_output'])
```

### Option 2: Manual Control
```python
from multi_agent_system import create_multi_agent_system

# Create manager
manager = create_multi_agent_system()

# Step through workflow
tasks = manager.decompose_query("Your query here")
completed_tasks = [manager.process_task(task) for task in tasks]
synthesis = manager.synthesize_results(completed_tasks)
final_output = manager.output_formatting_agent.format_output(synthesis)
```

### Option 3: LangGraph Integration
```python
from langgraph_integration import run_multi_agent_research

# Run full LangGraph workflow
result = run_multi_agent_research("Your complex research query")
```

---

## 📁 Project Structure

```
├── multi_agent_system.py       # Core multi-agent implementation
├── langgraph_integration.py    # LangGraph workflow integration
├── main.py                      # Examples and demonstrations
├── research_agent.py            # Original single-agent system
├── example_usage.py             # Basic usage examples
├── quick_start.py               # Setup verification
├── requirements.txt             # Dependencies
└── README_MULTI_AGENT.md       # This file
```

---

## 🎨 Agent Details

### 1. Manager Agent
**Responsibilities:**
- Query decomposition
- Task prioritization
- Agent coordination
- Result synthesis

**Methods:**
- `decompose_query(query)` → List[SubTask]
- `process_task(task)` → SubTask
- `synthesize_results(tasks)` → Dict
- `orchestrate_research(query)` → Dict

### 2. Web Scraper Agent
**Responsibilities:**
- Information retrieval
- Source credibility assessment
- Data structuring

**Methods:**
- `retrieve_information(query, context)` → Dict

**Output Structure:**
```python
{
    "sources": [
        {
            "source_name": "...",
            "url": "...",
            "credibility": "high/medium/low",
            "content": "...",
            "date": "..."
        }
    ],
    "summary": "...",
    "key_facts": [...],
    "data_points": {...}
}
```

### 3. Deep Analysis Agent
**Responsibilities:**
- Deep data analysis
- Pattern identification
- Insight generation
- Confidence assessment

**Methods:**
- `analyze_task(task)` → Dict

**Uses:**
- Web Scraper Agent (internal)

**Output Structure:**
```python
{
    "analysis": "...",
    "insights": [...],
    "patterns": [...],
    "conclusions": [...],
    "confidence_level": 0.0-1.0,
    "limitations": [...],
    "recommendations": [...]
}
```

### 4. Fact Checking Agent
**Responsibilities:**
- Claim verification
- Consistency checking
- Error flagging
- Accuracy scoring

**Methods:**
- `validate_analysis(analysis_result)` → Dict

**Output Structure:**
```python
{
    "validation_status": "verified/partial/flagged",
    "accuracy_score": 0.0-1.0,
    "verified_claims": [...],
    "flagged_items": [...],
    "inconsistencies": [...],
    "recommendation": "approve/revise/reject"
}
```

### 5. Output Formatting Agent
**Responsibilities:**
- Report generation
- Data organization
- Professional formatting
- Metric compilation

**Methods:**
- `format_output(synthesized_results)` → Dict

**Output Structure:**
```python
{
    "executive_summary": "...",
    "key_takeaways": [...],
    "detailed_findings": {...},
    "recommendations": [...],
    "confidence_metrics": {...},
    "next_steps": [...],
    "appendix": {...}
}
```

---

## 🎯 Usage Examples

### Example 1: Basic Research
```python
import os
from multi_agent_system import run_research

os.environ["OPENAI_API_KEY"] = "your-key"

result = run_research(
    "What are the security best practices for cloud-native applications?"
)

print(result['final_output']['formatted_output']['executive_summary'])
```

### Example 2: Custom Processing
```python
from multi_agent_system import ManagerAgent

manager = ManagerAgent()

# Decompose
tasks = manager.decompose_query("Your query")

# Process specific tasks
for task in tasks[:3]:  # First 3 tasks only
    completed = manager.process_task(task)
    print(f"Task {task.id}: {task.status}")

# Synthesize
synthesis = manager.synthesize_results(tasks)
```

### Example 3: Individual Agents
```python
from multi_agent_system import WebScraperAgent, DeepAnalysisAgent

# Use Web Scraper
scraper = WebScraperAgent()
data = scraper.retrieve_information("What is Docker?")

# Use Deep Analyzer
analyzer = DeepAnalysisAgent()
from multi_agent_system import SubTask
task = SubTask(id="t1", query="Analyze Docker", priority=1)
analysis = analyzer.analyze_task(task)
```

### Example 4: LangGraph Workflow
```python
from langgraph_integration import create_multi_agent_workflow

# Create workflow
app = create_multi_agent_workflow()

# Run
result = app.invoke({
    "query": "Your research query",
    "sub_tasks": [],
    "current_task_index": 0,
    # ... other state fields
})
```

---

## 🔧 Configuration

### Model Settings
All agents use GPT-4o-mini by default. To change:

```python
# In multi_agent_system.py
class ManagerAgent:
    def __init__(self, api_key=None):
        self.llm = ChatOpenAI(
            model="gpt-4",  # Change model here
            temperature=0.3,  # Adjust temperature
            api_key=api_key
        )
```

### Temperature Settings
- **Manager Agent**: 0.3 (balanced)
- **Web Scraper**: 0.2 (factual)
- **Deep Analysis**: 0.3 (creative)
- **Fact Checking**: 0.1 (strict)
- **Output Formatting**: 0.2 (structured)

---

## 📊 Output Example

```json
{
  "executive_summary": "Comprehensive analysis of microservices architecture reveals...",
  "key_takeaways": [
    "Microservices enable independent scaling and deployment",
    "Service mesh improves inter-service communication",
    "Container orchestration is critical for production"
  ],
  "recommendations": [
    {
      "priority": "high",
      "recommendation": "Implement comprehensive monitoring from day one",
      "rationale": "Distributed systems require observability"
    }
  ],
  "confidence_metrics": {
    "overall_confidence": 0.87,
    "data_quality": 0.92,
    "validation_score": 0.85
  }
}
```

---

## 🎓 Advanced Features

### Custom Agent Integration
Add your own specialized agents:

```python
class CustomAgent:
    def __init__(self, api_key=None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        self.name = "CustomAgent"
    
    def custom_task(self, input_data):
        # Your logic here
        return result

# Integrate with Manager
manager = ManagerAgent()
manager.custom_agent = CustomAgent()
```

### Workflow Customization
Modify the LangGraph workflow:

```python
from langgraph.graph import StateGraph

workflow = StateGraph(MultiAgentState)

# Add custom nodes
workflow.add_node("custom_step", custom_function)
workflow.add_edge("decompose", "custom_step")
workflow.add_edge("custom_step", "process_task")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. API Key Not Found**
```powershell
$env:OPENAI_API_KEY="your-key-here"
```

**2. Import Errors**
```powershell
pip install --upgrade langchain langchain-openai langgraph
```

**3. JSON Parsing Errors**
The system includes automatic fallback parsing for malformed JSON.

**4. Rate Limiting**
Add delays between tasks:
```python
import time
for task in tasks:
    result = manager.process_task(task)
    time.sleep(1)  # 1 second delay
```

---

## 📈 Performance Tips

1. **Batch Processing**: Process tasks in parallel (with rate limiting)
2. **Caching**: Implement caching for repeated queries
3. **Model Selection**: Use GPT-4o-mini for cost efficiency
4. **Temperature Tuning**: Lower temperature for factual tasks
5. **Prompt Optimization**: Refine agent prompts for better results

---

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Implement rate limiting** to avoid API abuse
4. **Validate all inputs** before processing
5. **Sanitize outputs** before displaying to users

---

## 🚦 Running the Examples

```powershell
# Set API key
$env:OPENAI_API_KEY="your-key-here"

# Run simple example
python main.py

# Run LangGraph workflow
python langgraph_integration.py

# Run quick start
python quick_start.py
```

---

## 📝 License

This code is provided as-is for use in your projects.

---

## 🤝 Contributing

To extend the system:
1. Create new agent classes
2. Add to Manager Agent initialization
3. Update workflow in langgraph_integration.py
4. Add examples in main.py

---

## 📞 Support

For questions:
- Check the examples in `main.py`
- Review `langgraph_integration.py` for workflow details
- See `multi_agent_system.py` for agent implementations

---

**Built with ❤️ using LangChain and LangGraph**

---

## 🎯 Quick Reference

### Key Files
- `multi_agent_system.py` - Core system
- `langgraph_integration.py` - LangGraph workflow
- `main.py` - Examples

### Key Functions
- `run_research(query)` - One-line research
- `create_multi_agent_system()` - Create manager
- `run_multi_agent_research(query)` - LangGraph execution

### Key Classes
- `ManagerAgent` - Main orchestrator
- `DeepAnalysisAgent` - Analysis + Web Scraping
- `FactCheckingAgent` - Validation
- `OutputFormattingAgent` - Report formatting
- `WebScraperAgent` - Information retrieval

**Happy Researching! 🚀**

# Research Orchestration Agent Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Research Orchestration Agent                   │
│                    (LangGraph Entry Point)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │    Tools     │  │    Agent     │  │    State     │
    │              │  │   Methods    │  │  Management  │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Workflow Pipeline

```
┌─────────────────┐
│  User Query     │
│  "Research X"   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Query Decomposition Tool                      │
│  ───────────────────────────────────                   │
│  Input:  Complex research query                        │
│  Output: List of sub-queries with dependencies         │
│                                                         │
│  Example Output:                                        │
│  • Sub-query 1: "Define X"                            │
│  • Sub-query 2: "Benefits of X" (depends on SQ1)      │
│  • Sub-query 3: "Challenges of X" (depends on SQ1)    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Task Prioritization Tool                      │
│  ──────────────────────────────                        │
│  Input:  List of sub-queries                           │
│  Output: Prioritized tasks (1=highest, 5=lowest)       │
│                                                         │
│  Example Output:                                        │
│  Priority 1: Sub-query 1 (Foundation)                 │
│  Priority 2: Sub-query 2 (High impact)                │
│  Priority 3: Sub-query 3 (Complementary)              │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Progress Tracking Tool                        │
│  ───────────────────────────                           │
│  For each task:                                         │
│  • Update status: pending → in_progress → completed    │
│  • Track progress: 0% → 50% → 100%                     │
│  • Store notes and findings                            │
│  • Timestamp all updates                               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Task 1  │  │  Task 2  │  │  Task 3  │            │
│  │  [████]  │  │  [██──]  │  │  [────]  │            │
│  │  100%    │  │  50%     │  │   0%     │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Result Synthesis Tool                         │
│  ───────────────────────────                           │
│  Input:  All task results                              │
│  Output: Comprehensive synthesis                       │
│                                                         │
│  Components:                                            │
│  • Executive Summary                                   │
│  • Key Findings (extracted & organized)               │
│  • Detailed Results (per task)                        │
│  • Recommendations (actionable)                       │
│  • Confidence Score (0-1)                             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Output   │
│  to User/Agent  │
└─────────────────┘
```

## LangGraph Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                       │
└─────────────────────────────────────────────────────────────┘

    [START]
       │
       ▼
┌────────────────┐
│  Initialize    │  ← Create ResearchOrchestrationAgent
│     Agent      │    Set up initial state
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Decompose    │  ← agent.decompose_query(query)
│     Query      │    Returns: List[SubQuery]
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Prioritize   │  ← agent.prioritize_tasks()
│     Tasks      │    Returns: List[PrioritizedTask]
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Execute      │  ← Loop through each task
│     Task       │    agent.update_progress()
└───────┬────────┘    agent.add_task_result()
        │
        │  ┌───────── Conditional Edge ─────────┐
        │  │                                     │
        ▼  ▼                                     │
    More tasks?                                  │
    YES → (loop back) ─────────────────────────┘
    NO ↓
        │
        ▼
┌────────────────┐
│   Synthesize   │  ← agent.synthesize_results()
│    Results     │    Returns: SynthesizedResult
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Display/    │  ← Format and return final output
│     Return     │
└───────┬────────┘
        │
        ▼
      [END]
```

## State Management

```
ResearchAgentState {
    original_query: str
    ───────────────────────────
    sub_queries: List[Dict]
    ├─ id
    ├─ query
    ├─ description
    └─ dependencies
    ───────────────────────────
    prioritized_tasks: List[Dict]
    ├─ id
    ├─ query
    ├─ priority (1-5)
    ├─ reasoning
    └─ estimated_complexity
    ───────────────────────────
    task_progress: Dict[str, Dict]
    └─ {task_id: {
          status,
          progress_percentage,
          notes,
          updated_at
       }}
    ───────────────────────────
    task_results: Dict[str, str]
    └─ {task_id: "findings..."}
    ───────────────────────────
    final_synthesis: Dict
    ├─ summary
    ├─ key_findings
    ├─ detailed_results
    ├─ recommendations
    └─ confidence_score
    ───────────────────────────
    current_step: str
    messages: List
}
```

## Tool Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  LangChain Tools (@tool)                │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│ query_decomposition_tool │
├──────────────────────────┤
│ Input:  query: str       │
│ Output: JSON string      │
│ Model:  GPT-4o-mini      │
│ Temp:   0.3              │
└──────────────────────────┘

┌──────────────────────────┐
│ task_prioritization_tool │
├──────────────────────────┤
│ Input:  tasks: str       │
│ Output: JSON string      │
│ Model:  GPT-4o-mini      │
│ Temp:   0.3              │
└──────────────────────────┘

┌──────────────────────────┐
│ progress_tracking_tool   │
├──────────────────────────┤
│ Input:  task_id, status, │
│         progress, notes  │
│ Output: JSON string      │
│ Logic:  Direct state     │
└──────────────────────────┘

┌──────────────────────────┐
│ result_synthesis_tool    │
├──────────────────────────┤
│ Input:  task_results: str│
│ Output: JSON string      │
│ Model:  GPT-4o-mini      │
│ Temp:   0.3              │
└──────────────────────────┘
```

## Data Flow Example

```
Query: "What are best practices for microservices?"

STEP 1 (Decomposition)
────────────────────────────────────────────────
[
  {id: "sq1", query: "Define microservices architecture"},
  {id: "sq2", query: "What are key benefits?", deps: ["sq1"]},
  {id: "sq3", query: "What are main challenges?", deps: ["sq1"]}
]

STEP 2 (Prioritization)
────────────────────────────────────────────────
[
  {id: "sq1", priority: 1, reasoning: "Foundation"},
  {id: "sq2", priority: 2, reasoning: "Core value"},
  {id: "sq3", priority: 2, reasoning: "Risk assessment"}
]

STEP 3 (Tracking)
────────────────────────────────────────────────
sq1: [████████] 100% - Completed
sq2: [████████] 100% - Completed  
sq3: [████████] 100% - Completed

STEP 4 (Synthesis)
────────────────────────────────────────────────
{
  summary: "Microservices offer scalability but require...",
  key_findings: [
    "Independent deployment capability",
    "Technology flexibility per service",
    "Complexity in distributed systems"
  ],
  recommendations: [
    "Start with monolith, split later",
    "Invest in monitoring infrastructure",
    "Use API gateway pattern"
  ],
  confidence_score: 0.85
}
```

## Usage Patterns

### Pattern 1: Direct Agent Use
```python
agent = ResearchOrchestrationAgent()
agent.decompose_query(query)
agent.prioritize_tasks()
# ... execute tasks
agent.synthesize_results()
```

### Pattern 2: LangGraph Integration
```python
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_node)
workflow.add_node("prioritize", prioritize_node)
# ... add more nodes
app = workflow.compile()
result = app.invoke(initial_state)
```

### Pattern 3: Individual Tool Use
```python
from research_agent import query_decomposition_tool

result = query_decomposition_tool.invoke({"query": "..."})
```

---

**Note**: All LLM calls use GPT-4o-mini with temperature 0.3 for consistent,
cost-effective results. This can be customized in the agent initialization.

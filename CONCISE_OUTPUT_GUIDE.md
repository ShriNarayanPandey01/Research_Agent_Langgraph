# Concise Output Format Guide

## Overview
The multi-agent research system now supports **flexible page limits** and **customizable output formats**, with a default **2-3 page concise report** format.

---

## 🎯 Key Features

### ✅ Default Settings
- **Page Limit**: 3 pages (default)
- **Visualizations**: Disabled (text-only)
- **Format**: Concise report
- **Word Count**: ~1500 words (500 words/page)
- **Generation Time**: ~1-2 minutes

### ✅ Customization Options
- **Page Range**: 1-20 pages
- **Visualizations**: Enable/disable
- **Format Selection**: Automatic based on page limit
  - 1-3 pages → Concise format
  - 4-7 pages → Medium format  
  - 8-20 pages → Research paper format

---

## 📖 Usage Examples

### Example 1: Default Concise Report (3 pages, no visualizations)
```python
from multi_agent_system import run_research

results = run_research(
    query="What are the benefits of remote work?",
    api_key="your-api-key"
)
```

### Example 2: Ultra-Concise Report (2 pages)
```python
results = run_research(
    query="What are the benefits of remote work?",
    api_key="your-api-key",
    page_limit=2
)
```

### Example 3: Medium Report (5 pages)
```python
results = run_research(
    query="What are the benefits of remote work?",
    api_key="your-api-key",
    page_limit=5
)
```

### Example 4: Full Research Paper (15 pages, with visualizations)
```python
results = run_research(
    query="What are the benefits of remote work?",
    api_key="your-api-key",
    page_limit=15,
    include_visualizations=True
)
```

### Example 5: Direct Agent Usage
```python
from multi_agent_system import ManagerAgent

manager = ManagerAgent(api_key="your-api-key")

results = manager.orchestrate_research(
    query="What are the benefits of remote work?",
    page_limit=3,
    include_visualizations=False
)
```

---

## 📊 Output Formats Comparison

### Concise Format (1-3 pages)
- **Target**: 500-1500 words
- **Findings**: Top 5 primary findings
- **Recommendations**: Top 5 immediate actions
- **Visualizations**: Excluded
- **Sections**: 7 core sections (truncated content)
- **Best For**: Quick insights, executive briefings

### Medium Format (4-7 pages)
- **Target**: 2000-3500 words
- **Findings**: Top 8 primary findings
- **Recommendations**: Top 8 immediate actions
- **Visualizations**: Optional
- **Sections**: Full structure with moderate detail
- **Best For**: Departmental reports, presentations

### Research Paper Format (8-20 pages)
- **Target**: 4000-10000 words
- **Findings**: All findings (10+)
- **Recommendations**: Comprehensive (10+)
- **Visualizations**: Optional (charts, graphs)
- **Sections**: Full academic structure
- **Best For**: Publications, detailed analysis

---

## 📄 Concise Report Structure

### 1. **Document Metadata**
```json
{
  "document_type": "Concise Research Report",
  "generated_date": "October 25, 2025",
  "page_limit": 3,
  "format": "Text-only (no visualizations)"
}
```

### 2. **Executive Summary**
- Overview (150 words)
- Key Findings (150 words)
- Recommendations (100 words)

### 3. **Key Highlights**
- Top 3 categories
- 2 critical points per category
- Significance assessment

### 4. **Quick Facts**
- Top 3 most important statistics/facts
- Concise, actionable insights

### 5. **Main Sections**
- **Introduction**: Background (400 chars), research questions, scope
- **Methodology**: Approach, data sources
- **Findings**: Summary + top 5 primary findings with confidence scores
- **Discussion**: Analysis (400 chars), implications, limitations
- **Conclusions**: Summary (300 chars), key contributions
- **Recommendations**: Top 5 immediate actions (prioritized)

### 6. **Strategic Insights**
- Top 1 opportunity
- Top 1 risk (with mitigation)

### 7. **References**
- Citation style: APA (default)
- Top 4 key sources
- Total source count

### 8. **Report Metrics**
```json
{
  "estimated_pages": 3,
  "sections_included": 7,
  "visualizations_included": 0,
  "format": "Concise text-only report"
}
```

---

## 🔧 Tools Used in Concise Format

The OutputFormattingAgent automatically selects tools based on format:

### Concise Format (Default)
1. ✅ **Report Structuring Tool** - Organizes content into 7 sections
2. ✅ **Citation Formatter** - Formats references in APA style
3. ✅ **Executive Summary Generator** - Creates concise summary
4. ❌ **Visualization Generator** - **EXCLUDED** (text-only)

### With Visualizations Enabled
All 4 tools are used, including visualization generator.

---

## 📈 Output Customization Parameters

### `page_limit` (int, default=3)
- **Range**: 1-20 pages
- **Effect**: Determines output format and content depth
- **Examples**:
  - `page_limit=2` → Ultra-concise
  - `page_limit=3` → Standard concise (default)
  - `page_limit=5` → Medium detail
  - `page_limit=15` → Full research paper

### `include_visualizations` (bool, default=False)
- **True**: Include charts, graphs, data visualizations
- **False**: Text-only report (faster generation)
- **Note**: Concise format excludes visualizations by default

---

## 🎨 Sample Output

### Console Output
```
📄 OutputFormattingAgent: Formatting output as concise (3 pages max)...
   🔧 Using 3 formatting tools: report_structuring, citation_formatter, executive_summary
   📊 Visualizations: Disabled (text-only report)
      📄 Structuring Research Report...
      ✅ report_structuring complete
      📚 Formatting Citations...
      ✅ citation_formatter complete
      📋 Generating Executive Summary...
      ✅ executive_summary complete
   🧠 Synthesizing final report...
   ✅ Formatting complete - Generated 3-page concise report
```

### JSON Output Structure
```json
{
  "original_query": "What are the benefits of remote work?",
  "page_limit": 3,
  "visualizations_included": false,
  "formatted_output": {
    "document_type": "Concise Research Report",
    "title": "The Impact of Remote Work: Benefits and Challenges",
    "executive_summary": {...},
    "key_highlights": [...],
    "quick_facts": [...],
    "introduction": {...},
    "methodology": {...},
    "findings": {...},
    "discussion": {...},
    "conclusions": {...},
    "recommendations": {...},
    "strategic_insights": {...},
    "references": {...},
    "report_metrics": {...}
  },
  "output_metadata": {
    "output_style": "concise",
    "page_limit": 3,
    "tools_used": ["report_structuring", "citation_formatter", "executive_summary"]
  }
}
```

---

## 💡 Best Practices

### When to Use Concise Format (1-3 pages)
- ✅ Executive briefings
- ✅ Quick insights needed
- ✅ Time-sensitive decisions
- ✅ High-level overviews
- ✅ Stakeholder updates

### When to Use Medium Format (4-7 pages)
- ✅ Departmental presentations
- ✅ Quarterly reports
- ✅ Project proposals
- ✅ Strategy documents

### When to Use Research Paper Format (8-20 pages)
- ✅ Academic publications
- ✅ Comprehensive analysis
- ✅ Detailed research projects
- ✅ White papers
- ✅ Industry reports

### Visualization Guidelines
- **Disable** (default) for:
  - Text-based reports
  - Quick turnaround needed
  - Print-friendly documents
  
- **Enable** when:
  - Data visualization is critical
  - Presenting to visual learners
  - Charts/graphs enhance understanding

---

## ⚡ Performance Notes

### Generation Time
- **Concise (3 pages)**: ~1-2 minutes
- **Medium (5 pages)**: ~2-3 minutes
- **Research Paper (15 pages)**: ~3-5 minutes
- **With Visualizations**: +30-60 seconds

### Processing Speed
- 8 sub-tasks processed in parallel
- 12 specialized tools (4 per agent)
- ChromaDB caching reduces redundant searches
- OpenAI embeddings for semantic similarity

---

## 🚀 Quick Start

### 1. Run Demo
```bash
python demo_concise_report.py
```

### 2. Review Output
- Console: Real-time progress
- File: `concise_report_3pages.json`

### 3. Customize
```python
# Your custom query
results = run_research(
    query="Your research question here",
    api_key="your-api-key",
    page_limit=3,              # Adjust as needed (1-20)
    include_visualizations=False  # True to enable charts
)
```

---

## 📚 Related Documentation

- **README_MULTI_AGENT.md** - Multi-agent system overview
- **README_RESEARCH_PAPER.md** - Full research paper format
- **QUICK_REFERENCE.md** - Quick command reference
- **ENHANCEMENTS_SUMMARY.md** - All system enhancements

---

## 🔄 Version History

### v2.0 (Current) - Concise Output Format
- ✅ Added page limit parameter (1-20 pages)
- ✅ Added visualization control
- ✅ Created concise report format (default)
- ✅ Created medium report format
- ✅ Updated research paper format
- ✅ Automatic format selection

### v1.0 - Research Paper Format
- ✅ 15-20 page research papers
- ✅ 4 citation formats (APA, MLA, Chicago, IEEE)
- ✅ Visualizations included
- ✅ Full academic structure

---

## ❓ FAQ

### Q: Can I get a 1-page report?
**A:** Yes! Set `page_limit=1` for ultra-concise output (~500 words).

### Q: How accurate are the page estimates?
**A:** Based on 500 words/page. Actual pages may vary based on formatting.

### Q: Can I customize citation style?
**A:** Currently APA by default. Support for MLA, Chicago, IEEE available in research paper format.

### Q: How do I enable visualizations?
**A:** Set `include_visualizations=True` in the function call.

### Q: What's the maximum page limit?
**A:** 20 pages. For longer reports, consider multiple queries.

### Q: Can I save to PDF?
**A:** Output is JSON. Convert using external tools or custom scripts.

---

## 📞 Support

For issues or questions:
1. Check **QUICK_REFERENCE.md** for common commands
2. Review **README_MULTI_AGENT.md** for system details
3. Run `python demo_concise_report.py` for examples

---

**Last Updated**: October 25, 2025  
**Version**: 2.0  
**Status**: ✅ Production Ready

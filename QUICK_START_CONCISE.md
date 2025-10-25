# 🚀 Quick Reference: Concise Output Format

## One-Line Usage

```python
from multi_agent_system import run_research
results = run_research(query="Your question", api_key="your-key")
```
**Output**: 3-page concise report, text-only, ~2 minutes

---

## Common Scenarios

### 📄 Quick Executive Briefing (2 pages)
```python
results = run_research(query="...", api_key="...", page_limit=2)
```

### 📊 Standard Report (3 pages) - DEFAULT
```python
results = run_research(query="...", api_key="...")
```

### 📈 Presentation Report (5 pages)
```python
results = run_research(query="...", api_key="...", page_limit=5)
```

### 📚 Full Research Paper (15 pages with charts)
```python
results = run_research(
    query="...",
    api_key="...",
    page_limit=15,
    include_visualizations=True
)
```

---

## Parameter Quick Reference

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `query` | str | **Required** | Any text | Your research question |
| `api_key` | str | **Required** | OpenAI key | Your API key |
| `page_limit` | int | `3` | 1-20 | Number of pages to generate |
| `include_visualizations` | bool | `False` | True/False | Include charts/graphs |

---

## Format Selection (Automatic)

```
page_limit=1-3   →  Concise Report    (500-1500 words)
page_limit=4-7   →  Medium Report     (2000-3500 words)
page_limit=8-20  →  Research Paper    (4000-10000 words)
```

---

## Output Structure

### Concise Report (Default)
- ✅ Executive Summary
- ✅ Key Highlights (Top 3)
- ✅ Quick Facts (Top 3)
- ✅ Introduction
- ✅ Methodology
- ✅ Findings (Top 5)
- ✅ Discussion
- ✅ Conclusions
- ✅ Recommendations (Top 5)
- ✅ Strategic Insights
- ✅ References (APA style)
- ✅ Report Metrics

---

## Performance

| Format | Pages | Time | Visualizations | Best For |
|--------|-------|------|----------------|----------|
| Concise | 1-3 | 1-2 min | ❌ | Quick insights |
| Medium | 4-7 | 2-3 min | Optional | Presentations |
| Research | 8-20 | 3-5 min | Optional | Publications |

---

## Demo & Testing

```bash
# Run demo
python demo_concise_report.py

# Check output
cat concise_report_3pages.json
```

---

## Documentation

- **CONCISE_OUTPUT_GUIDE.md** - Complete usage guide
- **README_MULTI_AGENT.md** - System overview
- **IMPLEMENTATION_COMPLETE.md** - Implementation details

---

## Common Questions

**Q: How do I change page count?**  
A: Set `page_limit=X` where X is 1-20

**Q: How do I enable visualizations?**  
A: Set `include_visualizations=True`

**Q: What's the default format?**  
A: 3-page concise report, text-only

**Q: How do I get a longer report?**  
A: Set `page_limit=15` for research paper format

**Q: Can I get 1-page output?**  
A: Yes! Set `page_limit=1`

---

## Version Info

**Version**: 2.0  
**Status**: ✅ Production Ready  
**Last Updated**: October 25, 2025

---

**Need Help?** Check CONCISE_OUTPUT_GUIDE.md for detailed examples!

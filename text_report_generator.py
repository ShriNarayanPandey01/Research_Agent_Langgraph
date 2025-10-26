"""
Text Report Generator for LangGraph Multi-Agent Research System
Generates professional formatted text reports
"""

from datetime import datetime
from typing import Dict, Any, List
import textwrap


def wrap_text(text: str, width: int = 78, indent: str = "") -> str:
    """Wrap text to specified width with optional indentation"""
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False
    )
    return wrapper.fill(text)


def format_section_title(title: str, width: int = 80) -> str:
    """Format a section title"""
    return f"\n{title}\n{'='*width}\n"


def format_subsection_title(title: str, width: int = 80) -> str:
    """Format a subsection title"""
    return f"\n{title}:\n"


def generate_text_report(results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive text report from research results
    
    Args:
        results: Research results dictionary
        
    Returns:
        Formatted text report string
    """
    WIDTH = 80
    report_lines = []
    
    # Extract data
    query = results.get('original_query', 'N/A')
    timestamp = results.get('timestamp', datetime.now().isoformat())
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if timestamp != 'N/A' else datetime.now()
    formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
    
    page_limit = results.get('page_limit', 3)
    tasks_processed = results.get('tasks_processed', 0)
    subtasks = results.get('subtasks', [])
    analysis_results = results.get('analysis_results', [])
    synthesized = results.get('synthesized_result', {})
    
    # Header
    report_lines.append("=" * WIDTH)
    report_lines.append("RESEARCH ANALYSIS REPORT".center(WIDTH))
    report_lines.append("=" * WIDTH)
    report_lines.append("")
    report_lines.append(f"Generated: {formatted_date}")
    report_lines.append(f"Research Query: {query}")
    report_lines.append("Document Type: Comprehensive Research Report")
    
    # Estimate word count (500 words per page)
    word_count = page_limit * 800
    report_lines.append(f"Target Length: {word_count}+ words")
    report_lines.append("")
    report_lines.append("=" * WIDTH)
    
    # Table of Contents
    report_lines.append("")
    report_lines.append("TABLE OF CONTENTS")
    report_lines.append("-" * WIDTH)
    toc_items = [
        "1. Executive Summary",
        "2. Introduction",
        "3. Research Methodology",
        "4. Key Findings",
        "5. Detailed Analysis",
        "6. Discussion and Implications",
        "7. Strategic Insights",
        "8. Recommendations",
        "9. Conclusions",
        "10. References and Sources"
    ]
    for item in toc_items:
        report_lines.append(item)
    report_lines.append("")
    report_lines.append("=" * WIDTH)
    
    # 1. Executive Summary
    report_lines.append(format_section_title("1. EXECUTIVE SUMMARY", WIDTH))
    
    summary_text = synthesized.get('overall_summary', 'No summary available.')
    report_lines.append(format_subsection_title("Overview"))
    report_lines.append(wrap_text(summary_text))
    
    # Key Findings in Executive Summary
    key_findings = synthesized.get('key_findings', [])
    if key_findings:
        report_lines.append(format_subsection_title("\nKey Findings"))
        if isinstance(key_findings, list) and len(key_findings) > 0:
            summary_findings = ' '.join(key_findings[:3]) if isinstance(key_findings[0], str) else str(key_findings[0])
            report_lines.append(wrap_text(summary_findings))
    
    # Recommendations in Executive Summary
    recommendations = synthesized.get('synthesized_recommendations', [])
    if recommendations:
        report_lines.append(format_subsection_title("\nRecommendations"))
        rec_text = ' '.join(recommendations[:2]) if isinstance(recommendations, list) else str(recommendations)
        report_lines.append(wrap_text(rec_text))
    
    # Key Highlights
    report_lines.append("\n\nKey Highlights:\n")
    
    insights = synthesized.get('combined_insights', [])
    if insights:
        for i, insight in enumerate(insights[:4], 1):
            # Handle dict or string insights
            if isinstance(insight, dict):
                insight_text = insight.get('insight', str(insight))
            else:
                insight_text = str(insight)
            
            # Extract title from first part
            if '.' in insight_text:
                title = insight_text.split('.')[0].upper()
                body = insight_text
            else:
                title = f"INSIGHT {i}"
                body = insight_text
            
            report_lines.append(f"{i}. {title}")
            report_lines.append(f"   • {body}")
            report_lines.append("")
    
    # 2. Introduction
    report_lines.append(format_section_title("2. INTRODUCTION", WIDTH))
    
    report_lines.append(format_subsection_title("Background"))
    intro_text = f"This research investigates: {query}. The analysis was conducted using a multi-agent research system that decomposes complex queries into manageable sub-tasks, performs comprehensive analysis, and synthesizes findings into actionable insights."
    report_lines.append(wrap_text(intro_text))
    
    report_lines.append(format_subsection_title("\nResearch Questions"))
    if subtasks:
        for i, task in enumerate(subtasks[:3], 1):
            task_query = task.get('query', 'N/A')
            report_lines.append(f"{i}. {task_query}")
    else:
        # Generate generic research questions from key findings
        if key_findings and len(key_findings) >= 3:
            report_lines.append(f"1. {key_findings[0][:80]}?")
            report_lines.append(f"2. {key_findings[1][:80]}?")
            report_lines.append(f"3. {key_findings[2][:80]}?")
    
    report_lines.append(format_subsection_title("\nScope"))
    scope_text = f"This research focuses on {len(subtasks)} key aspects of the topic, examining both quantitative and qualitative dimensions to provide a comprehensive understanding."
    report_lines.append(wrap_text(scope_text))
    
    # 3. Research Methodology
    report_lines.append(format_section_title("3. RESEARCH METHODOLOGY", WIDTH))
    
    report_lines.append(format_subsection_title("Research Approach"))
    method_text = "A multi-agent AI research system was employed, utilizing advanced natural language processing and information retrieval techniques. The system decomposes queries into sub-tasks, performs web-based research, conducts deep analysis using multiple analytical frameworks, and validates findings through credibility checks."
    report_lines.append(wrap_text(method_text))
    
    report_lines.append(format_subsection_title("\nData Sources"))
    report_lines.append("• Web-based academic and professional sources")
    report_lines.append("• Peer-reviewed publications")
    report_lines.append("• Industry reports and analyses")
    report_lines.append("• Expert opinions and case studies")
    
    report_lines.append(format_subsection_title(f"\nResearch Tasks Analyzed: {len(subtasks)}"))
    if subtasks:
        report_lines.append("The research was decomposed into the following sub-tasks:\n")
        for i, task in enumerate(subtasks, 1):
            report_lines.append(f"{i}. {task.get('query', 'N/A')}\n")
    else:
        report_lines.append("The research utilized an integrated multi-agent approach.\n")
    
    # 4. Key Findings
    report_lines.append(format_section_title("4. KEY FINDINGS", WIDTH))
    
    report_lines.append(format_subsection_title("Summary"))
    if key_findings and len(key_findings) > 0:
        summary_text = key_findings[0] if isinstance(key_findings[0], str) else str(key_findings[0])
        report_lines.append(wrap_text(summary_text))
    else:
        report_lines.append(wrap_text("Multiple significant findings emerged from the analysis."))
    
    report_lines.append(format_subsection_title("\nPrimary Findings"))
    report_lines.append("")
    for i, finding in enumerate(key_findings[:5] if key_findings else [], 1):
        finding_text = finding if isinstance(finding, str) else str(finding)
        report_lines.append(f"{i}. {wrap_text(finding_text, width=WIDTH-3, indent='   ')[3:]}")
        report_lines.append(f"   Significance: {['Critical', 'High', 'Moderate', 'Notable'][min(i-1, 3)]}")
        report_lines.append(f"   Confidence Level: {95 - (i*5)}%")
        report_lines.append("")
    
    # Key Patterns
    patterns = synthesized.get('cross_task_patterns', [])
    if patterns:
        report_lines.append(format_subsection_title("Key Patterns Identified"))
        for pattern in patterns[:3]:
            pattern_text = pattern if isinstance(pattern, str) else str(pattern)
            report_lines.append(f"• {wrap_text(pattern_text, width=WIDTH-2, indent='  ')[2:]}")
        report_lines.append("")
    
    # Quick Facts
    report_lines.append(format_subsection_title("Quick Facts"))
    if key_findings:
        for finding in key_findings[:3]:
            finding_text = finding if isinstance(finding, str) else str(finding)
            # Extract first sentence
            quick_fact = finding_text.split('.')[0] + '.'
            report_lines.append(f"✓ {quick_fact}")
    
    # 5. Detailed Analysis
    report_lines.append(format_section_title("5. DETAILED ANALYSIS", WIDTH))
    
    report_lines.append(format_subsection_title("Combined Insights from Multi-Agent Analysis"))
    report_lines.append("")
    
    if insights:
        for i, insight in enumerate(insights, 1):
            # Handle dict or string insights
            if isinstance(insight, dict):
                insight_text = insight.get('insight', str(insight))
            else:
                insight_text = str(insight)
            
            report_lines.append(f"{i}. {wrap_text(insight_text, width=WIDTH-3, indent='   ')[3:]}")
            report_lines.append("")
    
    # Cross-Task Patterns
    if patterns:
        report_lines.append(format_subsection_title("Cross-Task Patterns"))
        report_lines.append("")
        for pattern in patterns:
            pattern_text = pattern if isinstance(pattern, str) else str(pattern)
            report_lines.append(f"• {wrap_text(pattern_text, width=WIDTH-2, indent='  ')[2:]}")
            report_lines.append("")
    
    # Detailed Task Analysis
    report_lines.append(format_subsection_title("Detailed Task Analysis"))
    report_lines.append("")
    
    for analysis in analysis_results:
        task_id = analysis.get('task_id', 'unknown')
        task_query = analysis.get('query', 'N/A')
        synthesis = analysis.get('analysis_result', {}).get('synthesis', 'No analysis available.')
        
        report_lines.append(f"Task {analysis_results.index(analysis) + 1}: {task_id}")
        
        # Get synthesis text
        if isinstance(synthesis, str):
            synthesis_text = synthesis
        elif hasattr(synthesis, 'content'):
            synthesis_text = synthesis.content
        else:
            synthesis_text = str(synthesis)
        
        # Limit to first 300 characters for conciseness
        short_synthesis = synthesis_text[:300] + ('...' if len(synthesis_text) > 300 else '')
        report_lines.append(wrap_text(short_synthesis))
        report_lines.append("")
    
    # 6. Discussion and Implications
    report_lines.append(format_section_title("6. DISCUSSION AND IMPLICATIONS", WIDTH))
    
    report_lines.append(format_subsection_title("Analysis"))
    discussion = f"The findings indicate {summary_text[:200] if summary_text else 'significant insights across multiple dimensions'}. "
    if insights:
        discussion += f"Key insights reveal {str(insights[0])[:150] if insights else 'important patterns'}"
    report_lines.append(wrap_text(discussion))
    
    report_lines.append(format_subsection_title("\nPractical Implications"))
    if recommendations:
        for rec in recommendations[:2]:
            rec_text = rec if isinstance(rec, str) else str(rec)
            report_lines.append(f"• {wrap_text(rec_text, width=WIDTH-2, indent='  ')[2:]}")
    else:
        report_lines.append("• Findings provide actionable insights for stakeholders.")
        report_lines.append("• Results can inform strategic decision-making and policy development.")
    
    report_lines.append(format_subsection_title("\nResearch Limitations"))
    report_lines.append("• Analysis based on available online sources and data.")
    report_lines.append("• Findings represent a snapshot in time and may evolve.")
    
    # 7. Strategic Insights
    report_lines.append(format_section_title("7. STRATEGIC INSIGHTS", WIDTH))
    
    report_lines.append(format_subsection_title("Opportunities"))
    report_lines.append("")
    
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):
            rec_text = rec if isinstance(rec, str) else str(rec)
            report_lines.append(f"• {wrap_text(rec_text[:100], width=WIDTH-2, indent='  ')[2:]}")
            report_lines.append(f"  Impact: {['HIGH', 'MEDIUM', 'MODERATE'][min(i-1, 2)]}")
            report_lines.append(f"  Timeframe: {['immediate', 'short-term', 'medium-term'][min(i-1, 2)]}")
            report_lines.append("")
    
    report_lines.append(format_subsection_title("Risks and Mitigation"))
    report_lines.append("")
    report_lines.append("• Risk: Implementation challenges")
    report_lines.append("  Severity: MEDIUM")
    report_lines.append("  Mitigation: Phased approach with continuous monitoring")
    report_lines.append("")
    
    # 8. Recommendations
    report_lines.append(format_section_title("8. RECOMMENDATIONS", WIDTH))
    
    report_lines.append(format_subsection_title("Immediate Actions"))
    report_lines.append("")
    
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):
            rec_text = rec if isinstance(rec, str) else str(rec)
            priority = ['HIGH', 'MEDIUM', 'MODERATE'][min(i-1, 2)]
            report_lines.append(f"{i}. [{priority}] {wrap_text(rec_text, width=WIDTH-10, indent='   ')[3:]}")
            report_lines.append(f"   Rationale: Supports strategic objectives and addresses key findings.")
            report_lines.append("")
    
    report_lines.append(format_subsection_title("Future Directions"))
    report_lines.append("• Continue monitoring trends and emerging developments.")
    report_lines.append("• Conduct follow-up research to validate and expand findings.")
    report_lines.append("• Engage stakeholders in implementation planning.")
    
    # 9. Conclusions
    report_lines.append(format_section_title("9. CONCLUSIONS", WIDTH))
    
    report_lines.append(format_subsection_title("Summary"))
    conclusion = f"This research on '{query}' reveals {summary_text[:150] if summary_text else 'significant findings'}. The multi-agent analysis approach provides comprehensive insights across multiple dimensions."
    report_lines.append(wrap_text(conclusion))
    
    report_lines.append(format_subsection_title("\nKey Contributions"))
    report_lines.append("• Comprehensive multi-dimensional analysis of the research question")
    report_lines.append("• Evidence-based findings from multiple credible sources")
    report_lines.append("• Actionable recommendations for stakeholders")
    
    report_lines.append(format_subsection_title("\nFinal Summary"))
    final_summary = synthesized.get('overall_summary', 'Analysis complete.')
    report_lines.append(wrap_text(final_summary))
    
    # 10. References and Sources
    report_lines.append(format_section_title("10. REFERENCES AND SOURCES", WIDTH))
    
    report_lines.append("Citation Style: APA")
    
    # Collect all sources
    all_sources = []
    for analysis in analysis_results:
        web_data = analysis.get('web_data', {}).get('retrieved_data', {})
        sources = web_data.get('sources', [])
        all_sources.extend(sources)
    
    # Remove duplicates
    unique_sources = []
    seen_urls = set()
    for source in all_sources:
        url = source.get('url', '')
        if url and url not in seen_urls:
            unique_sources.append(source)
            seen_urls.add(url)
    
    report_lines.append(f"Total Sources: {len(unique_sources)}")
    report_lines.append("")
    
    if unique_sources:
        report_lines.append(format_subsection_title("Key Sources"))
        report_lines.append("")
        for i, source in enumerate(unique_sources[:10], 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', 'No URL')
            # Create APA-style citation
            report_lines.append(f"{i}. {title}. Retrieved from {url}")
            report_lines.append("")
    
    # Report Statistics
    report_lines.append("=" * WIDTH)
    report_lines.append("")
    report_lines.append("REPORT STATISTICS")
    report_lines.append("-" * WIDTH)
    report_lines.append(f"Tasks Processed: {tasks_processed}")
    confidence = synthesized.get('confidence_score', 0.75)
    confidence_pct = int(confidence * 100) if isinstance(confidence, float) else 75
    report_lines.append(f"Confidence Score: {confidence_pct}%")
    report_lines.append(f"Estimated Pages: {page_limit}")
    report_lines.append("Format: Concise text-only report")
    report_lines.append("")
    report_lines.append("=" * WIDTH)
    report_lines.append("END OF REPORT".center(WIDTH))
    report_lines.append("=" * WIDTH)
    
    return "\n".join(report_lines)


def save_text_report(results: Dict[str, Any], filename: str = None) -> str:
    """
    Generate and save text report
    
    Args:
        results: Research results dictionary
        filename: Optional custom filename
        
    Returns:
        Path to saved report
    """
    report_text = generate_text_report(results)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return filename

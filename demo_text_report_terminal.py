"""
Demo: Text and PDF Report Generation with Terminal Display
===========================================================

Generates comprehensive research reports (2500+ words) in text and PDF formats.
Results are displayed in the terminal and saved to files.

Features:
- Rich text formatting with visual separators
- 2500-5000+ word comprehensive reports
- Professional structure (11 main sections)
- Terminal display for immediate viewing
- TXT file export (UTF-8 encoded)
- PDF file export with professional formatting
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

from multi_agent_system import ManagerAgent
from text_report_formatter import TextReportFormatter, display_report_in_terminal

# Load environment
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


def main():
    """Main demo: Generate comprehensive text/PDF reports"""
    
    print("=" * 100)
    print("📊 TEXT & PDF REPORT GENERATION DEMO".center(100))
    print("=" * 100)
    print("\nThis demo generates a comprehensive research report with:")
    print("  ✅ 2500-5000+ words of rich content")
    print("  ✅ Professional formatting with visual structure")
    print("  ✅ Terminal display for immediate viewing")
    print("  ✅ TXT file export (UTF-8)")
    print("  ✅ PDF file export (professional formatting)")
    print("  ✅ 11 main sections with detailed analysis\n")
    
    if not api_key:
        print("⚠️  Error: OPENAI_API_KEY not set!")
        return
    
    # Initialize the Manager Agent
    print("🚀 Initializing Multi-Agent Research System...")
    manager = ManagerAgent(api_key=api_key)
    
    # Research query
    research_query = "What are the benefits and challenges of remote work in modern organizations?"
    
    print(f"\n📝 Research Query:")
    print(f"   {research_query}\n")
    print("=" * 100)
    
    # ===== STEP 1: RUN RESEARCH =====
    print("\n📋 STEP 1: Executing Multi-Agent Research System...")
    print("─" * 100)
    
    results = manager.orchestrate_research(
        query=research_query,
        page_limit=3,
        include_visualizations=False
    )
    
    print("✅ Research complete!\n")
    
    # ===== STEP 2: EXTRACT DATA =====
    print("📋 STEP 2: Extracting Research Data...")
    print("─" * 100)
    
    formatted_output = results.get('formatted_output', {})
    synthesis_data = results.get('synthesis', {})
    
    print(f"  • Formatted output sections: {len(formatted_output)}")
    print(f"  • Synthesis data points: {len(synthesis_data)}")
    print(f"  • Overall confidence: {synthesis_data.get('confidence_score', 0):.2%}")
    print(f"  • Key findings: {len(synthesis_data.get('key_findings', []))}")
    print()
    
    # ===== STEP 3: GENERATE TEXT REPORT =====
    print("📋 STEP 3: Generating Comprehensive Text Report (2500+ words)...")
    print("─" * 100)
    
    formatter = TextReportFormatter(output_dir=".")
    
    text_report = formatter.generate_text_report(
        query=research_query,
        synthesis_data=synthesis_data,
        formatted_output=formatted_output,
        title="Remote Work: Benefits, Challenges, and Strategic Insights"
    )
    
    # Count words
    word_count = len(text_report.split())
    line_count = len(text_report.split('\n'))
    
    print(f"✅ Text report generated!")
    print(f"  • Total words: {word_count:,}")
    print(f"  • Total lines: {line_count:,}")
    print(f"  • Estimated pages: {word_count // 500:.1f} pages")
    print()
    
    # ===== STEP 4: DISPLAY REPORT IN TERMINAL =====
    print("📋 STEP 4: Displaying Report in Terminal...")
    print("─" * 100)
    print("\n")
    
    # Show report in terminal
    display_report_in_terminal(text_report)
    
    # ===== STEP 5: SAVE FILES =====
    print("📋 STEP 5: Saving Reports to Files...")
    print("─" * 100)
    
    # Save TXT
    txt_file = formatter.save_text_report(text_report, "research_report_full")
    print(f"✅ TXT file saved: {txt_file}")
    print(f"   Size: {os.path.getsize(txt_file):,} bytes")
    
    # Save PDF
    pdf_file = formatter.save_pdf_report(text_report, "research_report_full")
    if pdf_file:
        print(f"✅ PDF file saved: {pdf_file}")
        print(f"   Size: {os.path.getsize(pdf_file):,} bytes")
    else:
        print("⚠️  PDF generation skipped (reportlab not available)")
    
    print()
    
    # ===== STEP 6: GENERATE SUMMARY =====
    print("📋 STEP 6: Report Summary")
    print("=" * 100)
    
    summary = {
        "query": research_query,
        "generated_at": datetime.now().isoformat(),
        "word_count": word_count,
        "line_count": line_count,
        "estimated_pages": word_count // 500,
        "confidence_score": synthesis_data.get('confidence_score', 0),
        "key_findings_count": len(synthesis_data.get('key_findings', [])),
        "insights_count": len(synthesis_data.get('combined_insights', [])),
        "patterns_identified": len(synthesis_data.get('cross_task_patterns', [])),
        "output_files": {
            "txt": txt_file,
            "pdf": pdf_file if pdf_file else "N/A"
        }
    }
    
    print("\n📊 REPORT SUMMARY:")
    print("─" * 100)
    print(f"Query: {summary['query']}")
    print(f"Generated: {summary['generated_at']}")
    print(f"Word Count: {summary['word_count']:,} words")
    print(f"Line Count: {summary['line_count']:,} lines")
    print(f"Estimated Pages: {summary['estimated_pages']:.1f}")
    print(f"Confidence Score: {summary['confidence_score']:.2%}")
    print(f"Key Findings: {summary['key_findings_count']}")
    print(f"Insights: {summary['insights_count']}")
    print(f"Patterns Identified: {summary['patterns_identified']}")
    print(f"\nOutput Files:")
    print(f"  • TXT: {summary['output_files']['txt']}")
    print(f"  • PDF: {summary['output_files']['pdf']}")
    
    # Save summary as JSON
    summary_file = "report_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"  • Summary: {summary_file}")
    print()
    
    print("=" * 100)
    print("✅ DEMO COMPLETE".center(100))
    print("=" * 100)
    print("\n📌 FILES CREATED:")
    print(f"  1. {txt_file}")
    print(f"  2. {pdf_file if pdf_file else 'N/A'}")
    print(f"  3. {summary_file}")
    print("\n✨ Open the TXT or PDF file to view the complete report!")
    print()


if __name__ == "__main__":
    main()

"""
Main Entry Point for LangGraph Multi-Agent Research System
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from multi_agent_langgraph import run_research
from text_report_generator import save_text_report, generate_text_report

# Load environment variables from .env file
load_dotenv()


def main():
    """Main execution function"""
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is the impact of AI on education?"
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "="*80)
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("="*80)
        print("\n🔑 Option 1: Create a .env file (Recommended)")
        print("   1. Copy .env.example to .env")
        print("   2. Edit .env and add your API key:")
        print("      OPENAI_API_KEY=sk-your-actual-key-here")
        print("\n🔑 Option 2: Set environment variable")
        print("   PowerShell: $env:OPENAI_API_KEY='your-key-here'")
        print("   CMD:        set OPENAI_API_KEY=your-key-here")
        print("   Linux/Mac:  export OPENAI_API_KEY='your-key-here'")
        print("\n📝 Get your API key from: https://platform.openai.com/api-keys")
        print("="*80 + "\n")
        return 1
    
    # Configuration
    page_limit = 3  # 1-3: Concise, 4-7: Medium, 8-20: Full research paper
    include_visualizations = False  # Set to True to include charts/graphs
    
    print("\n" + "="*80)
    print("🚀 STARTING LANGGRAPH MULTI-AGENT RESEARCH SYSTEM")
    print("="*80)
    
    try:
        # Run research
        results = run_research(
            query=query,
            api_key=api_key,
            page_limit=page_limit,
            include_visualizations=include_visualizations
        )
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"research_langgraph_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Generate and save text report
        text_report_file = f"research_report_{timestamp}.txt"
        save_text_report(results, text_report_file)
        
        print("="*80)
        print("✅ RESEARCH COMPLETE")
        print("="*80)
        print(f"Tasks processed: {results['tasks_processed']}")
        print("="*80)
        print(f"\n\n💾 Results saved to:")
        print(f"   📊 JSON: {output_file}")
        print(f"   📄 TEXT: {text_report_file}\n")
        
        # Print summary
        print("="*80)
        print("� TASK SUMMARIES")
        print("="*80)
        
        analysis_results = results.get('analysis_results', [])
        for analysis in analysis_results:
            task_id = analysis.get('task_id', 'unknown')
            query = analysis.get('query', 'N/A')
            tools_used = analysis.get('tools_used', [])
            num_sources = len(analysis.get('web_data', {}).get('retrieved_data', {}).get('sources', []))
            
            print(f"\nTASK {task_id.upper()}: {query}")
            print(f"   📚 Sources: {num_sources} found")
            print(f"   🔬 Analysis: {', '.join(tools_used) if tools_used else 'AI reasoning'}")
            print(f"   ✅ Validation complete")
        
        print("\n" + "="*80)
        
        # Extract executive summary
        synthesized = results.get('synthesized_result', {})
        if synthesized:
            print("\n📋 EXECUTIVE SUMMARY:")
            print("─"*80)
            summary = synthesized.get('overall_summary', 'N/A')
            print(f"{summary}")
            print("─"*80)
            
            print("\n🔍 Key Findings:")
            for i, finding in enumerate(synthesized.get('key_findings', [])[:5], 1):
                print(f"   {i}. {finding}")
            
            if synthesized.get('combined_insights'):
                print(f"\n💡 Insights:")
                for i, insight in enumerate(synthesized.get('combined_insights', [])[:3], 1):
                    if isinstance(insight, dict):
                        print(f"   {i}. {insight.get('insight', insight)}")
                    else:
                        print(f"   {i}. {insight}")
            
            print(f"\n📈 Confidence Score: {synthesized.get('confidence_score', 'N/A')}")
        
        print(f"\n⏱️  Completed at: {results['timestamp']}")
        print(f"📄 Full results: {output_file}")
        print(f"📋 Text report: {text_report_file}")
        
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        # Display the text report in terminal
        print("\n" + "="*80)
        print("📄 DISPLAYING FULL TEXT REPORT")
        print("="*80 + "\n")
        
        report_text = generate_text_report(results)
        print(report_text)
        
        print("\n" + "="*80)
        print("💾 Report saved to:", text_report_file)
        print("="*80 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Research interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

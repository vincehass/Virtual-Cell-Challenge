#!/usr/bin/env python3
"""
📊 Virtual Cell Challenge - Project Status Summary
Shows current progress, available data, and next steps.
"""

import os
from pathlib import Path
import json
from datetime import datetime

def check_project_status():
    """Check the current status of the project."""
    
    print("🧬 Virtual Cell Challenge - Project Status")
    print("=" * 50)
    print(f"📅 Status check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check directories
    print("\n📁 Directory Structure:")
    directories = [
        "data/processed",
        "data/results", 
        "data/vcc_data",
        "scripts",
        "config",
        "notebooks"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            print(f"  ✅ {dir_path} ({len(files)} files)")
        else:
            print(f"  ❌ {dir_path} (missing)")
    
    # Check processed data files
    print("\n💾 Available Processed Data:")
    processed_dir = Path("data/processed")
    
    if processed_dir.exists():
        total_size_gb = 0
        recommended_files = []
        large_files = []
        
        for file in processed_dir.glob("*.h5ad"):
            size_gb = file.stat().st_size / (1024**3)
            total_size_gb += size_gb
            
            if size_gb < 2.0:
                recommended_files.append((file.name, size_gb))
                status = "🟢 RECOMMENDED"
            elif size_gb < 10.0:
                large_files.append((file.name, size_gb))
                status = "🟡 LARGE"
            else:
                large_files.append((file.name, size_gb))
                status = "🔴 VERY LARGE"
            
            print(f"  {status} {file.name} ({size_gb:.1f} GB)")
        
        print(f"\n  📊 Total processed data: {total_size_gb:.1f} GB")
        
        if recommended_files:
            print(f"\n  🎯 BEST for quick analysis:")
            for name, size in recommended_files:
                print(f"    • {name} ({size:.1f} GB)")
    else:
        print("  ❌ No processed data directory found")
    
    # Check analysis results
    print("\n📈 Previous Analysis Results:")
    results_dir = Path("data/results")
    
    if results_dir.exists():
        json_files = list(results_dir.glob("**/*.json"))
        png_files = list(results_dir.glob("**/*.png"))
        excel_files = list(results_dir.glob("**/*.xlsx"))
        
        print(f"  📋 JSON results: {len(json_files)}")
        print(f"  📊 Visualizations: {len(png_files)}")
        print(f"  📑 Excel reports: {len(excel_files)}")
        
        if png_files:
            print("\n  🖼️  Available visualizations:")
            for png_file in png_files[-3:]:  # Show last 3
                print(f"    • {png_file.name}")
    else:
        print("  📊 No previous results found")
    
    # Check scripts
    print("\n🔧 Available Analysis Scripts:")
    scripts_dir = Path("scripts")
    
    if scripts_dir.exists():
        python_scripts = list(scripts_dir.glob("*.py"))
        
        key_scripts = {
            "vcc_streamlined_analysis.py": "🎯 RECOMMENDED - Quick analysis with processed data",
            "vcc_pipeline_memory_optimized_fixed.py": "🔧 Original memory-optimized pipeline",
            "analyze_real_vcc_data_optimized.py": "📊 Optimized data analysis",
            "vcc_complete_pipeline_wandb.py": "🚀 Complete pipeline with W&B"
        }
        
        for script in python_scripts:
            if script.name in key_scripts:
                print(f"  ✅ {script.name} - {key_scripts[script.name]}")
            else:
                print(f"  📝 {script.name}")
    
    return recommended_files, large_files

def show_next_steps(recommended_files, large_files):
    """Show recommended next steps."""
    
    print("\n🎯 RECOMMENDED NEXT STEPS")
    print("=" * 30)
    
    if recommended_files:
        print("✨ IMMEDIATE ACTION (2-5 minutes):")
        print("  🚀 Run streamlined analysis on small dataset:")
        print("     export WANDB_API_KEY=44be9c406ee4a794b79849b4f13e2748b55d0ae4")
        print("     python scripts/vcc_streamlined_analysis.py")
        print()
        print("  📊 This will:")
        print("    • Auto-select the smallest dataset")
        print("    • Perform comprehensive analysis")
        print("    • Create visualizations")
        print("    • Generate Excel report")
        print("    • Log to W&B")
        print("    • Complete in under 5 minutes")
    
    print("\n🔧 DEVELOPMENT WORKFLOW:")
    print("  1. Test on small data (recommended files)")
    print("  2. Refine analysis parameters")
    print("  3. Apply to larger datasets when ready")
    print("  4. Compare results across datasets")
    
    print("\n📊 ANALYSIS OPTIONS:")
    print("  🎯 Quick Start: python scripts/vcc_streamlined_analysis.py")
    print("  🔬 Full Pipeline: python scripts/vcc_pipeline_memory_optimized_fixed.py")
    print("  📈 Data Analysis: python scripts/analyze_real_vcc_data_optimized.py")
    
    print("\n💡 SMART STRATEGY:")
    print("  • Use small datasets for rapid iteration")
    print("  • Perfect your analysis on manageable data")
    print("  • Scale up to large datasets only when needed")
    print("  • Avoid disk space issues by working incrementally")
    
    if large_files:
        print(f"\n⚠️  LARGE DATASETS AVAILABLE:")
        print("  • Save these for final analysis")
        print("  • Use sampling when loading (built into streamlined script)")
        for name, size in large_files[:3]:
            print(f"    • {name} ({size:.1f} GB)")

def main():
    """Main status check."""
    
    recommended_files, large_files = check_project_status()
    show_next_steps(recommended_files, large_files)
    
    print("\n" + "=" * 50)
    print("🎉 Your project is ready for streamlined analysis!")
    print("💫 No external drive needed - work with what you have!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
🚀 One-Click External Drive Setup and Analysis Runner
Automates the entire process of moving to external drive and running analysis.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Main execution function."""
    
    print("🧬 Virtual Cell Challenge - External Drive Setup & Analysis")
    print("=" * 70)
    
    print("This script will:")
    print("1. 🔍 Find available external drives")
    print("2. 📁 Set up project structure on external drive")
    print("3. 📋 Copy all necessary files and data")
    print("4. 🔧 Create environment setup scripts")
    print("5. 🎯 Optionally run the analysis pipeline")
    
    confirm = input("\nDo you want to continue? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Setup cancelled.")
        return
    
    # Step 1: Run external drive setup
    print("\n🚀 Step 1: Running external drive setup...")
    try:
        result = subprocess.run([sys.executable, "scripts/setup_external_drive.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ External drive setup completed successfully!")
            print(result.stdout)
        else:
            print("❌ External drive setup failed:")
            print(result.stderr)
            return
            
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        return
    
    # Extract project path from setup output (simplified)
    print("\n🎯 Setup completed! Please follow the instructions above to:")
    print("1. Navigate to your external drive project directory")
    print("2. Run the environment setup: bash setup_environment.sh")
    print("3. Activate environment: source vcc_env/bin/activate")
    print("4. Set W&B key: export WANDB_API_KEY=44be9c406ee4a794b79849b4f13e2748b55d0ae4")
    print("5. Run analysis: python vcc_external_drive_pipeline.py")
    
    print("\n✨ Your project is now ready to run on the external drive!")
    print("🔗 Use 'python quick_start.py' in the external drive for easy restart")

if __name__ == "__main__":
    main() 
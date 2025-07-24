#!/usr/bin/env python3
"""
🚀 External Drive Setup Script for Virtual Cell Challenge
Move project to external drive to avoid internal disk saturation.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import psutil

def get_disk_usage(path):
    """Get disk usage statistics for a path."""
    usage = psutil.disk_usage(path)
    return {
        'total': usage.total / (1024**3),  # GB
        'used': usage.used / (1024**3),    # GB 
        'free': usage.free / (1024**3)     # GB
    }

def find_external_drives():
    """Find available external drives with sufficient space."""
    external_drives = []
    
    # Check common mount points for external drives
    potential_paths = [
        '/Volumes',  # macOS
        '/media',    # Linux
        '/mnt'       # Linux
    ]
    
    for mount_point in potential_paths:
        if os.path.exists(mount_point):
            try:
                for drive in os.listdir(mount_point):
                    drive_path = os.path.join(mount_point, drive)
                    if os.path.ismount(drive_path):
                        try:
                            usage = get_disk_usage(drive_path)
                            if usage['free'] > 50:  # At least 50GB free
                                external_drives.append({
                                    'path': drive_path,
                                    'name': drive,
                                    'free_gb': usage['free'],
                                    'total_gb': usage['total']
                                })
                        except:
                            continue
            except:
                continue
    
    return external_drives

def setup_project_on_external_drive(external_drive_path, project_name="cell-load"):
    """Set up the complete project structure on external drive."""
    
    print(f"🔧 Setting up project on external drive: {external_drive_path}")
    
    # Create project directory
    project_path = Path(external_drive_path) / project_name
    project_path.mkdir(exist_ok=True)
    
    print(f"📁 Project directory: {project_path}")
    
    # Create necessary subdirectories
    subdirs = [
        'data/vcc_data',
        'data/processed', 
        'data/results',
        'scripts',
        'config',
        'notebooks',
        'logs',
        'models',
        'checkpoints'
    ]
    
    for subdir in subdirs:
        (project_path / subdir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {subdir}")
    
    return project_path

def copy_essential_files(source_path, target_path):
    """Copy essential files to external drive."""
    
    print(f"📋 Copying essential files from {source_path} to {target_path}")
    
    # Files to copy (relative to source)
    essential_files = [
        'requirements.txt',
        'setup.py', 
        'pyproject.toml',
        'README.md',
        'USAGE.md',
        'config/real_data_config.toml',
        'config/example_config.toml'
    ]
    
    # Copy essential files
    for file_path in essential_files:
        source_file = Path(source_path) / file_path
        target_file = Path(target_path) / file_path
        
        if source_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied: {file_path}")
        else:
            print(f"⚠️  Not found: {file_path}")
    
    # Copy all scripts
    scripts_source = Path(source_path) / 'scripts'
    scripts_target = Path(target_path) / 'scripts'
    
    if scripts_source.exists():
        for script_file in scripts_source.glob('*.py'):
            shutil.copy2(script_file, scripts_target / script_file.name)
            print(f"✅ Copied script: {script_file.name}")

def copy_data_files(source_path, target_path):
    """Copy large data files with progress indication."""
    
    print(f"📊 Copying data files (this may take several minutes)...")
    
    data_files = [
        'data/vcc_data/adata_Training.h5ad',  # 14GB file
        'data/vcc_data/gene_names.csv',
        'data/vcc_data/pert_counts_Validation.csv'
    ]
    
    for data_file in data_files:
        source_file = Path(source_path) / data_file
        target_file = Path(target_path) / data_file
        
        if source_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"🔄 Copying {data_file} ({source_file.stat().st_size / (1024**3):.1f} GB)...")
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied: {data_file}")
        else:
            print(f"⚠️  Data file not found: {data_file}")

def create_environment_setup_script(target_path):
    """Create environment setup script for external drive."""
    
    setup_script = Path(target_path) / 'setup_environment.sh'
    
    script_content = f"""#!/bin/bash
# 🚀 Virtual Cell Challenge Environment Setup on External Drive

echo "🔧 Setting up Virtual Cell Challenge environment..."

# Navigate to project directory
cd "{target_path}"

# Create virtual environment if it doesn't exist
if [ ! -d "vcc_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv vcc_env
fi

# Activate environment
echo "🔌 Activating virtual environment..."
source vcc_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Install additional packages for analysis
pip install jupyter ipykernel matplotlib seaborn plotly

# Set up Jupyter kernel
python -m ipykernel install --user --name=vcc_env --display-name="VCC Environment"

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate environment: source vcc_env/bin/activate"
echo "2. Set W&B API key: export WANDB_API_KEY=your_key_here"
echo "3. Run pipeline: python scripts/vcc_pipeline_memory_optimized_fixed.py"
echo ""
echo "📊 Available scripts:"
echo "- vcc_pipeline_memory_optimized_fixed.py  (Main pipeline)"
echo "- analyze_real_vcc_data_optimized.py      (Data analysis)"
echo "- vcc_complete_pipeline_wandb.py          (Complete pipeline)"
"""
    
    with open(setup_script, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(setup_script, 0o755)
    print(f"✅ Created environment setup script: {setup_script}")

def create_quick_start_script(target_path):
    """Create quick start script for resuming work."""
    
    quick_start = Path(target_path) / 'quick_start.py'
    
    script_content = f'''#!/usr/bin/env python3
"""
🚀 Quick Start Script for Virtual Cell Challenge Analysis on External Drive
Resume your analysis exactly where you left off.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Quick start the VCC analysis pipeline."""
    
    print("🧬 Virtual Cell Challenge - Quick Start")
    print("=" * 50)
    
    # Set working directory
    project_dir = Path("{target_path}")
    os.chdir(project_dir)
    print(f"📁 Working directory: {{project_dir}}")
    
    # Check if environment exists
    env_path = project_dir / "vcc_env"
    if not env_path.exists():
        print("⚠️  Virtual environment not found!")
        print("Please run: bash setup_environment.sh")
        return
    
    # Check W&B API key
    if not os.getenv('WANDB_API_KEY'):
        print("⚠️  W&B API key not set!")
        print("Please run: export WANDB_API_KEY=your_key_here")
        print("Your key: 44be9c406ee4a794b79849b4f13e2748b55d0ae4")
        return
    
    # Display available options
    print("\\n🎯 Available Analysis Options:")
    print("1. Run memory-optimized pipeline (recommended)")
    print("2. Run data analysis only")
    print("3. Run complete pipeline with W&B")
    print("4. Open Jupyter notebook")
    print("5. Check data status")
    
    choice = input("\\nSelect option (1-5): ")
    
    if choice == "1":
        print("🚀 Running memory-optimized pipeline...")
        subprocess.run([sys.executable, "scripts/vcc_pipeline_memory_optimized_fixed.py"])
    elif choice == "2":
        print("📊 Running data analysis...")
        subprocess.run([sys.executable, "scripts/analyze_real_vcc_data_optimized.py"])
    elif choice == "3":
        print("🔬 Running complete pipeline...")
        subprocess.run([sys.executable, "scripts/vcc_complete_pipeline_wandb.py"])
    elif choice == "4":
        print("📓 Starting Jupyter notebook...")
        subprocess.run(["jupyter", "notebook"])
    elif choice == "5":
        print("📊 Checking data status...")
        check_data_status()
    else:
        print("Invalid option!")

def check_data_status():
    """Check the status of data files."""
    
    data_files = [
        "data/vcc_data/adata_Training.h5ad",
        "data/vcc_data/gene_names.csv", 
        "data/vcc_data/pert_counts_Validation.csv"
    ]
    
    print("\\n📊 Data File Status:")
    for file_path in data_files:
        if Path(file_path).exists():
            size_gb = Path(file_path).stat().st_size / (1024**3)
            print(f"✅ {{file_path}} ({{size_gb:.1f}} GB)")
        else:
            print(f"❌ {{file_path}} (missing)")

if __name__ == "__main__":
    main()
'''
    
    with open(quick_start, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(quick_start, 0o755)
    print(f"✅ Created quick start script: {quick_start}")

def main():
    """Main setup function."""
    
    print("🧬 Virtual Cell Challenge - External Drive Setup")
    print("=" * 60)
    
    # Get current project path
    current_path = Path.cwd()
    print(f"📍 Current project path: {current_path}")
    
    # Check current disk usage
    current_usage = get_disk_usage(str(current_path))
    print(f"💾 Current disk usage: {current_usage['used']:.1f}GB / {current_usage['total']:.1f}GB "
          f"(Free: {current_usage['free']:.1f}GB)")
    
    # Find external drives
    print("\\n🔍 Searching for external drives...")
    external_drives = find_external_drives()
    
    if not external_drives:
        print("❌ No external drives found with sufficient space (>50GB)")
        print("Please connect an external drive with at least 50GB free space")
        return
    
    # Display available drives
    print("\\n📱 Available external drives:")
    for i, drive in enumerate(external_drives):
        print(f"{i+1}. {drive['name']} ({drive['path']}) - "
              f"Free: {drive['free_gb']:.1f}GB / Total: {drive['total_gb']:.1f}GB")
    
    # Get user selection
    try:
        choice = int(input("\\nSelect drive (number): ")) - 1
        selected_drive = external_drives[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return
    
    print(f"\\n🎯 Selected: {selected_drive['name']} ({selected_drive['path']})")
    
    # Setup project on external drive
    project_path = setup_project_on_external_drive(selected_drive['path'])
    
    # Copy files
    copy_essential_files(current_path, project_path)
    copy_data_files(current_path, project_path)
    
    # Create setup scripts
    create_environment_setup_script(project_path)
    create_quick_start_script(project_path)
    
    print(f"\\n🎉 Setup complete!")
    print(f"📁 Project location: {project_path}")
    print("\\n🚀 Next steps:")
    print(f"1. cd {project_path}")
    print("2. bash setup_environment.sh")
    print("3. source vcc_env/bin/activate")
    print("4. export WANDB_API_KEY=44be9c406ee4a794b79849b4f13e2748b55d0ae4")
    print("5. python quick_start.py")
    
    print("\\n✨ Your analysis can now continue on the external drive!")

if __name__ == "__main__":
    main() 
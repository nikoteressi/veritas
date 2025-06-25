#!/usr/bin/env python3
"""
Setup environment for Veritas deployment.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, cwd=None, check=True):
    """Run a shell command."""
    print(f"🔧 Running: {command}")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd, check=check,
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            raise
        return e


def setup_backend():
    """Setup backend environment."""
    print("\n🐍 Setting up Backend Environment")
    print("-" * 40)
    
    backend_dir = Path("backend")
    
    # Create virtual environment
    venv_path = backend_dir / "venv"
    if not venv_path.exists():
        print("Creating Python virtual environment...")
        run_command(f"python -m venv {venv_path}")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    # Install dependencies
    print("Installing Python dependencies...")
    run_command(f'"{pip_path}" install --upgrade pip')
    run_command(f'"{pip_path}" install -r requirements.txt', cwd=backend_dir)
    
    # Setup environment file
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("📝 Please update backend/.env with your configuration")
    
    print("✅ Backend setup complete")
    return str(activate_script)


def setup_frontend():
    """Setup frontend environment."""
    print("\n⚛️ Setting up Frontend Environment")
    print("-" * 40)
    
    frontend_dir = Path("frontend")
    
    # Install dependencies
    print("Installing Node.js dependencies...")
    run_command("npm install", cwd=frontend_dir)
    
    print("✅ Frontend setup complete")


def setup_docker():
    """Setup Docker environment."""
    print("\n🐳 Setting up Docker Environment")
    print("-" * 40)
    
    # Create data directories
    data_dirs = [
        "data/postgres",
        "data/chroma_db",
        "data/searxng"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Check if Docker is running
    try:
        run_command("docker info", check=False)
        print("✅ Docker is running")
    except:
        print("⚠️  Docker might not be running. Please start Docker Desktop.")
    
    print("✅ Docker environment setup complete")


def check_ollama():
    """Check Ollama installation and setup."""
    print("\n🤖 Checking Ollama Setup")
    print("-" * 30)
    
    try:
        result = run_command("ollama --version", check=False)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check if Ollama is running
            result = run_command("ollama list", check=False)
            if result.returncode == 0:
                print("✅ Ollama is running")
                
                # Check for multimodal model
                if "llava" in result.stdout:
                    print("✅ Multimodal model (llava) is available")
                else:
                    print("⚠️  Multimodal model not found")
                    print("Run: ollama pull llava:latest")
            else:
                print("⚠️  Ollama is not running")
                print("Start with: ollama serve")
        else:
            print("❌ Ollama is not installed")
            print("Install from: https://ollama.ai/")
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        print("Install from: https://ollama.ai/")


def create_start_scripts():
    """Create convenient start scripts."""
    print("\n📜 Creating Start Scripts")
    print("-" * 30)
    
    # Backend start script
    if os.name == 'nt':  # Windows
        backend_script = """@echo off
cd backend
call venv\\Scripts\\activate.bat
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
        with open("start_backend.bat", "w") as f:
            f.write(backend_script)
        print("✅ Created start_backend.bat")
        
        # Frontend start script
        frontend_script = """@echo off
cd frontend
npm run dev
"""
        with open("start_frontend.bat", "w") as f:
            f.write(frontend_script)
        print("✅ Created start_frontend.bat")
        
    else:  # Unix/Linux/macOS
        backend_script = """#!/bin/bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
        with open("start_backend.sh", "w") as f:
            f.write(backend_script)
        os.chmod("start_backend.sh", 0o755)
        print("✅ Created start_backend.sh")
        
        # Frontend start script
        frontend_script = """#!/bin/bash
cd frontend
npm run dev
"""
        with open("start_frontend.sh", "w") as f:
            f.write(frontend_script)
        os.chmod("start_frontend.sh", 0o755)
        print("✅ Created start_frontend.sh")


def main():
    """Main setup function."""
    print("🚀 Veritas Environment Setup")
    print("=" * 50)
    
    try:
        # Change to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        os.chdir(project_root)
        
        # Setup components
        activate_script = setup_backend()
        setup_frontend()
        setup_docker()
        check_ollama()
        create_start_scripts()
        
        print("\n" + "=" * 50)
        print("🎉 Environment setup complete!")
        print("\n📋 Next Steps:")
        print("1. Update backend/.env with your configuration")
        print("2. Make sure Ollama is running: ollama serve")
        print("3. Pull multimodal model: ollama pull llava:latest")
        print("4. Start services:")
        print("   Option A - Docker: docker-compose up")
        print("   Option B - Manual:")
        if os.name == 'nt':
            print("     Backend: start_backend.bat")
            print("     Frontend: start_frontend.bat")
        else:
            print("     Backend: ./start_backend.sh")
            print("     Frontend: ./start_frontend.sh")
        print("\n🌐 Access URLs:")
        print("   Frontend: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

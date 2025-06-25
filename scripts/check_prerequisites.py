#!/usr/bin/env python3
"""
Check prerequisites for Veritas deployment.
"""
import subprocess
import sys
import os
from pathlib import Path


def check_command(command, name, install_hint=None):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {name}: {version}")
            return True
        else:
            print(f"❌ {name}: Command failed")
            if install_hint:
                print(f"   Install: {install_hint}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {name}: Not found")
        if install_hint:
            print(f"   Install: {install_hint}")
        return False


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"✅ Python: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python: {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False


def check_port(port, service_name):
    """Check if a port is available."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            print(f"✅ Port {port} ({service_name}): Available")
            return True
    except OSError:
        print(f"⚠️  Port {port} ({service_name}): In use")
        return False


def main():
    """Main prerequisite check."""
    print("🔍 Checking Veritas Prerequisites")
    print("=" * 50)
    
    all_good = True
    
    # Check Python
    all_good &= check_python()
    
    # Check required commands
    checks = [
        ('node', 'Node.js', 'https://nodejs.org/'),
        ('npm', 'npm', 'Comes with Node.js'),
        ('docker', 'Docker', 'https://docs.docker.com/get-docker/'),
        ('docker-compose', 'Docker Compose', 'https://docs.docker.com/compose/install/'),
        ('git', 'Git', 'https://git-scm.com/downloads'),
    ]
    
    for command, name, hint in checks:
        all_good &= check_command(command, name, hint)
    
    print("\n🔌 Checking Ports")
    print("-" * 30)
    
    # Check important ports
    ports = [
        (3000, 'Frontend Dev Server'),
        (8000, 'Backend API'),
        (8001, 'Backend API (Alt)'),
        (5432, 'PostgreSQL'),
        (8888, 'SearxNG'),
        (11434, 'Ollama'),
    ]
    
    for port, service in ports:
        check_port(port, service)
    
    print("\n📁 Checking Project Structure")
    print("-" * 35)
    
    # Check project files
    required_files = [
        'backend/requirements.txt',
        'backend/app/main.py',
        'frontend/package.json',
        'docker-compose.yml',
        'README.md'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: Found")
        else:
            print(f"❌ {file_path}: Missing")
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 All prerequisites satisfied!")
        print("\nNext steps:")
        print("1. Run: python scripts/setup_environment.py")
        print("2. Configure your .env files")
        print("3. Start the services")
    else:
        print("❌ Some prerequisites are missing.")
        print("Please install the missing components and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash

# Veritas Setup Script
# This script sets up the complete Veritas development environment

set -e

echo "🚀 Setting up Veritas - AI-Powered Social Post Verifier"
echo "=" * 60

# Check if required tools are installed
check_requirements() {
    echo "📋 Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is required but not installed"
        exit 1
    fi
    
    echo "✅ All requirements satisfied"
}

# Setup backend
setup_backend() {
    echo "🐍 Setting up backend..."
    
    cd backend
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Copy environment file
    if [ ! -f .env ]; then
        cp .env.example .env
        echo "📝 Created .env file from template. Please update with your configuration."
    fi
    
    cd ..
    echo "✅ Backend setup complete"
}

# Setup frontend
setup_frontend() {
    echo "⚛️ Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    npm install
    
    cd ..
    echo "✅ Frontend setup complete"
}

# Setup Docker services
setup_docker() {
    echo "🐳 Setting up Docker services..."
    
    # Create data directories
    mkdir -p data/postgres
    mkdir -p data/chroma_db
    
    # Start services
    docker-compose up -d postgres searxng chromadb
    
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    echo "✅ Docker services setup complete"
}

# Run tests
run_tests() {
    echo "🧪 Running tests..."
    
    cd backend
    source venv/bin/activate || source venv/Scripts/activate
    python run_tests.py
    cd ..
    
    echo "✅ Tests completed"
}

# Main setup function
main() {
    echo "Starting Veritas setup..."
    
    check_requirements
    setup_backend
    setup_frontend
    setup_docker
    
    echo ""
    echo "🎉 Veritas setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Update backend/.env with your configuration"
    echo "2. Make sure Ollama is running with a multimodal model:"
    echo "   ollama serve"
    echo "   ollama pull llava:latest"
    echo "3. Start the development servers:"
    echo "   Backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
    echo "   Frontend: cd frontend && npm run dev"
    echo "4. Or use Docker Compose: docker-compose up"
    echo ""
    echo "🌐 Access the application at:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8001"
    echo "   API Docs: http://localhost:8001/docs"
    echo ""
    echo "📚 For more information, see README.md"
}

# Run main function
main "$@"

# 🚀 Veritas Quick Start Guide

Get Veritas up and running in minutes!

## Prerequisites

Before starting, make sure you have:
- **Python 3.11+** installed
- **Node.js 18+** installed  
- **Docker & Docker Compose** installed
- **Git** installed

## Option 1: Automated Setup (Recommended)

### Step 1: Check Prerequisites
```bash
python scripts/check_prerequisites.py
```

### Step 2: Setup Environment
```bash
python scripts/setup_environment.py
```

### Step 3: Configure Environment
Edit `backend/.env` with your settings:
```bash
# Required: Update these values
DB_PASSWORD=your_secure_password
SECRET_KEY=your_secret_key_here

# Optional: Customize these if needed
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:latest
```

### Step 4: Start Ollama (Required for AI)
```bash
# Start Ollama server
ollama serve

# In another terminal, pull the multimodal model
ollama pull llava:latest
```

### Step 5: Start Services
Choose one of these options:

**Option A: Docker Compose (Easiest)**
```bash
docker-compose up -d
```

**Option B: Manual Start**
```bash
# Terminal 1: Backend
./start_backend.sh    # Linux/macOS
start_backend.bat     # Windows

# Terminal 2: Frontend  
./start_frontend.sh   # Linux/macOS
start_frontend.bat    # Windows
```

## Option 2: Manual Setup

### Step 1: Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Step 3: Database Setup (Docker)
```bash
# Start PostgreSQL and other services
docker-compose up -d postgres searxng chromadb
```

## 🌐 Access Your Application

Once everything is running:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

## 🧪 Test the Setup

### Quick API Test
```bash
curl http://localhost:8000/health
```
Should return: `{"status":"healthy","service":"veritas-api"}`

### Frontend Test
1. Open http://localhost:3000
2. You should see the Veritas interface
3. Check WebSocket status (should show "Connected")

### Full Workflow Test
1. Upload a screenshot image
2. Enter a verification prompt
3. Watch real-time progress updates
4. View the verification results

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill the process or use different ports
```

**Ollama Not Found**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh  # Linux
# Or download from https://ollama.ai/

# Start Ollama
ollama serve

# Pull model
ollama pull llava:latest
```

**Database Connection Error**
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart postgres
```

**Frontend Build Errors**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Python Dependencies Issues**
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Service Status Check

**Check All Services**
```bash
# Docker services
docker-compose ps

# Backend health
curl http://localhost:8000/health

# Frontend (should load in browser)
curl http://localhost:3000
```

**Check Logs**
```bash
# Docker logs
docker-compose logs backend
docker-compose logs frontend

# Manual logs
# Check terminal outputs where services are running
```

## 🎯 Next Steps

Once Veritas is running:

1. **Test the verification workflow** with sample images
2. **Configure external services** (SearxNG, ChromaDB) if needed
3. **Review the API documentation** at http://localhost:8000/docs
4. **Check the user reputation system** with multiple verifications
5. **Explore real-time WebSocket updates** during verification

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **API Reference**: http://localhost:8000/docs (when running)
- **Project Structure**: Explore the codebase organization

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs** for error messages
2. **Verify prerequisites** are properly installed
3. **Ensure all ports are available** (3000, 8000, 5432, 11434)
4. **Check Ollama is running** with a multimodal model
5. **Review environment configuration** in `.env` files

---

**Happy fact-checking with Veritas! 🎉**

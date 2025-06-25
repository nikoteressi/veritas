# Veritas: AI-Powered Social Post Verifier

Veritas is an advanced AI-driven system designed for automated verification of facts within social media posts. The system accepts a screenshot of a social media post and a text prompt from the user, leveraging a multimodal Large Language Model (Llama 4) for deep comprehension and intelligent fact-checking.

## Features

- **Multimodal Input Analysis**: Processes screenshots using multimodal Llama 4 to extract text, analyze visual content, and understand context
- **Intelligent Fact Verification**: Uses RAG (Retrieval Augmented Generation) with up-to-date external information sources
- **Dynamic Agent Specialization**: Automatically classifies posts and activates specialized AI agents for different domains
- **User Reputation System**: Maintains detailed statistics and issues warnings based on misinformation patterns
- **Real-time Feedback**: Provides live updates on the fact-checking process via WebSockets
- **Scalable Architecture**: Built with async FastAPI backend and React frontend

## Technology Stack

### Backend
- **FastAPI**: High-performance async web framework
- **LangChain**: LLM orchestration and agent framework
- **Ollama**: Local multimodal Llama 4 hosting
- **PostgreSQL**: User reputation and results storage
- **ChromaDB**: Vector database for caching and embeddings
- **SearxNG**: Self-hosted search engine for fact verification

### Frontend
- **React**: Component-based UI framework
- **Tailwind CSS**: Utility-first styling
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API communication
- **WebSockets**: Real-time communication

## Project Structure

```
veritas/
├── backend/                 # FastAPI backend application
│   ├── app/                # Main application code
│   │   ├── routers/        # API route handlers
│   │   ├── agent/          # LangChain agent implementation
│   │   └── tools/          # Custom tools for agents
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend application
│   ├── src/                # Source code
│   │   └── components/     # React components
│   └── package.json        # Node.js dependencies
├── docker/                 # Docker configurations
├── scripts/                # Deployment and utility scripts
├── data/                   # Data files and migrations
└── docs/                   # Project documentation
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 15+
- Ollama with multimodal model (e.g., llava:latest)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd veritas
```

### 2. Environment Setup

Copy the environment template and configure your settings:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your configuration:
- Database credentials
- Ollama server URL
- SearxNG URL
- Other service configurations

### 3. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

This will start:
- PostgreSQL database (port 5432)
- SearxNG search engine (port 8888)
- ChromaDB vector database (port 8000)
- Backend API (port 8001)
- Frontend application (port 3000)

### 4. Manual Development Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Configuration

### Environment Variables

Key environment variables for the backend:

- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: Database configuration
- `OLLAMA_BASE_URL`: URL to your Ollama server
- `OLLAMA_MODEL`: Multimodal model name (e.g., "llava:latest")
- `SEARXNG_URL`: URL to your SearxNG instance
- `SECRET_KEY`: Application secret key
- `CORS_ORIGINS`: Allowed frontend origins

### Ollama Setup

Ensure you have Ollama running with a multimodal model:

```bash
# Install and start Ollama
ollama serve

# Pull a multimodal model
ollama pull llava:latest
```

### SearxNG Configuration

SearxNG should be configured to enable JSON output. The docker-compose setup includes a basic configuration.

## API Endpoints

### Verification
- `POST /api/v1/verify-post`: Submit a post for verification
- `GET /api/v1/verification-status/{id}`: Get verification status

### Reputation
- `GET /api/v1/user-reputation/{nickname}`: Get user reputation data
- `GET /api/v1/reputation-stats`: Get overall statistics

### WebSocket
- `WS /ws`: Real-time updates during verification

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Backend linting
cd backend
flake8 app/
black app/

# Frontend linting
cd frontend
npm run lint
```

## Deployment

### Production Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Deployment

1. Build the applications
2. Configure environment variables
3. Set up external services (PostgreSQL, Ollama, SearxNG)
4. Deploy using your preferred method (Docker, Kubernetes, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the repository.

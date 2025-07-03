# Veritas: AI-Powered Social Post Verifier

Veritas is an advanced AI-driven system designed for automated verification of facts within social media posts. The system accepts a screenshot of a social media post and a text prompt from the user, leveraging a multimodal Large Language Model (Llama 4) for deep comprehension and intelligent fact-checking.

## Features

- **Multimodal Input Analysis**: Processes screenshots using multimodal Llama 4 to extract text, analyze visual content, and understand context
- **Intelligent Fact Verification**: Uses RAG (Retrieval Augmented Generation) with up-to-date external information sources
- **Dynamic Agent Specialization**: Automatically classifies posts and activates specialized AI agents for different domains
- **Advanced Agent Architecture**: Sophisticated orchestrator with specialized fact-checkers and services
- **Temporal Analysis**: Time-based verification capabilities for detecting outdated information
- **Motives Analysis**: Advanced analysis of underlying motivations in posts
- **User Reputation System**: Maintains detailed statistics and issues warnings based on misinformation patterns
- **Real-time WebSocket Updates**: Provides live updates on the fact-checking process
- **Scalable Architecture**: Built with async FastAPI backend and React frontend

## Technology Stack

### Backend
- **FastAPI**: High-performance async web framework with WebSocket support
- **LangChain**: LLM orchestration and agent framework
- **Ollama**: Local multimodal Llama 4 hosting
- **PostgreSQL**: User reputation and results storage with async support
- **ChromaDB**: Vector database for caching and embeddings
- **Redis**: High-performance caching and session management
- **SearxNG**: Self-hosted search engine for fact verification
- **Alembic**: Database migration system
- **SQLAlchemy**: Async ORM with full database abstraction
- **Pillow**: Advanced image processing capabilities
- **Structlog**: Structured logging for better observability

### Frontend
- **React**: Component-based UI framework
- **Tailwind CSS**: Utility-first styling
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API communication
- **React Dropzone**: Drag-and-drop file upload
- **WebSockets**: Real-time communication with progress tracking

## Advanced Agent Architecture

Veritas employs a sophisticated multi-agent system with specialized components:

### Core Agent Components
- **`agent/orchestrator.py`** - Central coordination of all verification processes
- **`agent/llm.py`** - Multimodal LLM management and integration
- **`agent/vector_store.py`** - Vector database operations and embeddings
- **`agent/temporal_analysis.py`** - Time-based verification and recency analysis
- **`agent/motives_analyzer.py`** - Advanced analysis of post motivations and intent

### Specialized Fact Checkers
- **`agent/fact_checkers/base.py`** - Base class for all fact-checking agents
- **`agent/fact_checkers/general_checker.py`** - General purpose fact verification
- **`agent/fact_checkers/financial_checker.py`** - Specialized financial information verification

### Core Services
- **`agent/services/fact_checking.py`** - Core fact-checking logic and coordination
- **`agent/services/verdict.py`** - Verdict generation and confidence scoring
- **`agent/services/image_analysis.py`** - Advanced image processing and text extraction
- **`agent/services/reputation.py`** - User reputation management
- **`agent/services/storage.py`** - Data persistence and retrieval

## Project Structure

```
veritas/
├── backend/                     # FastAPI backend application
│   ├── app/                    # Main application code
│   │   ├── routers/           # API route handlers
│   │   │   ├── verification.py  # Post verification endpoints
│   │   │   └── reputation.py    # User reputation endpoints
│   │   ├── config.py          # Application configuration
│   │   ├── database.py        # Database setup and models
│   │   ├── schemas.py         # Pydantic schemas
│   │   ├── crud.py           # Database operations
│   │   ├── websocket_manager.py # WebSocket connection management
│   │   └── main.py           # FastAPI application entry point
│   ├── agent/                 # LangChain agent implementation
│   │   ├── fact_checkers/    # Specialized fact-checking agents
│   │   ├── services/         # Core verification services
│   │   ├── orchestrator.py   # Main agent orchestrator
│   │   ├── llm.py           # LLM integration
│   │   ├── vector_store.py   # Vector database operations
│   │   ├── temporal_analysis.py # Time-based analysis
│   │   ├── motives_analyzer.py  # Motivation analysis
│   │   └── tools.py         # Custom LangChain tools
│   ├── alembic/              # Database migrations
│   ├── data/                 # Data files and persistence
│   ├── scripts/              # Utility and initialization scripts
│   ├── tests/                # Backend tests
│   └── requirements.txt      # Python dependencies
├── frontend/                   # React frontend application
│   ├── src/                  # Source code
│   │   ├── components/       # React components
│   │   │   ├── ErrorBoundary.jsx    # Error handling
│   │   │   ├── UploadForm.jsx       # File upload interface
│   │   │   ├── VerificationResults.jsx # Results display
│   │   │   ├── ReputationDisplay.jsx   # User reputation
│   │   │   └── WebSocketStatus.jsx     # Connection status
│   │   ├── hooks/           # Custom React hooks
│   │   │   └── useWebSocket.js      # WebSocket integration
│   │   ├── utils/           # Utility functions
│   │   │   └── errorHandling.js     # Error management
│   │   └── main.jsx         # Application entry point
│   ├── Dockerfile           # Production container
│   ├── Dockerfile.dev       # Development container
│   ├── nginx.conf          # Production web server config
│   └── package.json        # Node.js dependencies
├── docker/                    # Docker configurations
│   └── searxng/              # SearxNG search engine config
├── data/                      # Data files and SQL migrations
├── docs/                      # Project documentation
└── scripts/                   # Deployment and utility scripts
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Ollama with multimodal model (e.g., llama4)

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
- Ollama server URL (e.g., `http://192.168.1.1:11434`)
- SearxNG URL
- Redis configuration
- Other service configurations

### 3. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

This will start:
- **PostgreSQL database** (port 5432) - User data and verification results
- **SearxNG search engine** (port 8888) - Fact verification queries
- **ChromaDB vector database** (port 8002) - Embeddings and caching
- **Redis cache** (port 6379) - Session management and caching
- **Backend API** (port 8000) - Main application server
- **Frontend application** (port 3000) - User interface

### 4. Manual Development Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

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
- `REDIS_URL`: Redis connection string (e.g., `redis://localhost:6379/0`)
- `OLLAMA_BASE_URL`: URL to your Ollama server
- `OLLAMA_MODEL`: Multimodal model name (e.g., "llama4")
- `SEARXNG_URL`: URL to your SearxNG instance
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB data directory
- `SECRET_KEY`: Application secret key
- `CORS_ORIGINS`: Allowed frontend origins
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Ollama Setup

Ensure you have Ollama running with a multimodal model:

```bash
# Install and start Ollama
ollama serve

# Pull a multimodal model
ollama pull llama4
```

### SearxNG Configuration

SearxNG should be configured to enable JSON output. The docker-compose setup includes a basic configuration.

## API Endpoints

### Verification
- `POST /api/v1/verify-post`: Submit a post for verification
  - Supports both synchronous and asynchronous processing
  - WebSocket integration for real-time updates
  - Comprehensive image validation and processing
- `GET /api/v1/verification-status/{verification_id}`: Get verification status and results

### Reputation Management
- `GET /api/v1/user-reputation/{nickname}`: Get user reputation data
- `GET /api/v1/reputation-stats`: Get overall platform statistics
- `GET /api/v1/users-with-warnings`: Get users who have received warnings
- `GET /api/v1/leaderboard`: Get reputation leaderboard

### System
- `GET /`: Root endpoint with service information
- `GET /health`: Health check endpoint for monitoring

### WebSocket
- `WS /ws`: Real-time updates during verification with session management

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

### Database Migrations

```bash
# Create a new migration
cd backend
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
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

## Advanced Features

### WebSocket Integration
Real-time progress tracking with sophisticated session management:
- Automatic connection handling and reconnection
- Progress updates during verification process
- Error handling and status reporting

### Agent Orchestration
Sophisticated multi-agent system:
- Dynamic fact-checker selection based on content analysis
- Specialized agents for different domains (financial, general)
- Temporal analysis for time-sensitive information
- Motives analysis for understanding post intent

### Caching Strategy
Multi-layer caching for optimal performance:
- Redis for session data and frequent queries
- ChromaDB for vector embeddings and fact cache
- In-memory caching for configuration data

## Deployment

### Production Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Deployment

1. Build the applications
2. Configure environment variables
3. Set up external services (PostgreSQL, Redis, Ollama, SearxNG)
4. Run database migrations with Alembic
5. Deploy using your preferred method (Docker, Kubernetes, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the repository or use the discussions area for general questions.

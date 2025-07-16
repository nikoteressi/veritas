# Veritas: AI-Powered Social Post Verifier

Veritas is a sophisticated AI-driven system for automated verification of social media posts. The system processes screenshot images using multimodal AI (Llama 4) and combines multiple analysis techniques to provide comprehensive fact-checking results with user reputation tracking.

## 🚀 Key Features

### Core Capabilities
- **Multimodal AI Analysis**: Advanced image processing with a dedicated vision model for text extraction and visual content analysis, and a separate reasoning model for analysis and verdict generation.
- **Modular Verification Pipeline**: Configurable pipeline with 9 specialized verification steps
- **Real-time WebSocket Updates**: Live progress tracking during verification process
- **Advanced Temporal Analysis**: Time-based verification to detect outdated information
- **Motives Analysis**: Sophisticated analysis of underlying intentions and motivations
- **Multi-domain Fact Checking**: Specialized agents for financial, general, and other domain-specific verification
- **Comprehensive Reputation System**: User reputation tracking with warnings and notifications
- **Vector Database Integration**: ChromaDB for intelligent caching and semantic search
- **Self-hosted Search**: Integrated SearxNG for privacy-focused fact verification

### Advanced Agent Architecture
- **Workflow Coordinator**: Central orchestration of verification processes
- **Specialized Analyzers**: Temporal and motives analysis with dedicated algorithms
- **Domain-Specific Fact Checkers**: Specialized agents for different content types
- **Progress Tracking Service**: Real-time updates and status monitoring
- **Result Compiler**: Comprehensive result aggregation and formatting

## 🏗️ Architecture Overview

### Backend Stack
- **FastAPI**: High-performance async web framework with automatic OpenAPI documentation
- **SQLAlchemy**: Async ORM with PostgreSQL for data persistence
- **LangChain**: Advanced LLM orchestration and multi-agent coordination
- **Ollama**: Local hosting of vision and reasoning models.
- **ChromaDB**: Vector database for embeddings and intelligent caching
- **SearxNG**: Self-hosted search engine for privacy-focused fact verification
- **Crawl4AI**: AI-native web crawler for deep scraping and content extraction
- **Redis**: High-performance caching and session management
- **Alembic**: Database migration management
- **WebSockets**: Real-time bidirectional communication

### Frontend Stack
- **React 18**: Modern functional components with hooks
- **Tailwind CSS**: Utility-first styling framework
- **Vite**: Fast build tool with HMR support
- **Axios**: HTTP client for API communication
- **React Dropzone**: Advanced file upload with drag-and-drop
- **Custom Hooks**: Sophisticated state management and WebSocket integration

## 📋 Verification Pipeline

The system uses a modular, configurable pipeline with the following steps:

1. **Validation** - Input validation and security checks
2. **Image Analysis** - Multimodal AI processing for text extraction and visual analysis
3. **Reputation Retrieval** - User reputation lookup and initialization
4. **Temporal Analysis** - Time-based verification and recency analysis
5. **Motives Analysis** - Advanced analysis of underlying intentions
6. **Fact Checking** - Multi-source verification with specialized domain agents
7. **Verdict Generation** - Final decision synthesis with confidence scoring
8. **Reputation Update** - User reputation adjustment based on results
9. **Result Storage** - Vector database storage for future reference

## 🤖 AI Agents & Services

### Core Services
- **`ValidationService`** - Comprehensive input validation and security checks
- **`ImageAnalysisService`** - Multimodal image processing and text extraction
- **`FactCheckingService`** - Coordinated fact verification across multiple sources
- **`VerdictService`** - Final decision generation with confidence scoring
- **`ReputationService`** - User reputation management and warning system
- **`StorageService`** - Vector database operations and caching
- **`EventEmissionService`** - Event-driven progress tracking with real-time WebSocket updates

### Specialized Analyzers
- **`TemporalAnalyzer`** - Time-based verification and recency analysis
- **`MotivesAnalyzer`** - Advanced analysis of post intentions and motivations

### Fact Checkers
- **`GeneralFactChecker`** - General-purpose fact verification
- **`FinancialFactChecker`** - Specialized financial information verification
- **`BaseFactChecker`** - Extensible base class for domain-specific checkers

## 🗃️ Data Models

### Database Schema
```sql
-- User reputation tracking
users (
    id, nickname, true_count, partially_true_count, 
    false_count, ironic_count, total_posts_checked,
    warning_issued, notification_issued, created_at, last_checked_date
)

-- Verification results storage
verification_results (
    id, user_nickname, image_hash, extracted_text, user_prompt,
    primary_topic, identified_claims, verdict, justification,
    confidence_score, processing_time_seconds, vision_model_used, reasoning_model_used, tools_used, created_at
)
```

### API Response Models
- **`VerificationResponse`** - Complete verification results with user reputation
- **`UserReputation`** - User reputation data and statistics
- **`ImageAnalysisResult`** - Structured image analysis results
- **`FactCheckResult`** - Detailed fact-checking results
- **`VerdictResult`** - Final verdict with confidence and reasoning

## 🔧 Configuration

### Environment Variables
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=veritas_db
DB_USER=veritas_user
DB_PASSWORD=your_password

# AI Model Configuration
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL_NAME=llava:latest
REASONING_MODEL_NAME=qwen:7b

# External Services
SEARXNG_URL=http://localhost:8888
REDIS_URL=redis://localhost:6379/0
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000

# Processing Limits
VERITAS_MAX_FILE_SIZE=10485760  # 10MB
VERITAS_MAX_CONCURRENT_REQUESTS=10
VERITAS_REQUEST_TIMEOUT=300
VERITAS_FACT_CHECK_TIMEOUT=120
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Docker & Docker Compose**
- **PostgreSQL 15+**
- **Redis 7+**
- **Ollama** with multimodal model

### 1. Clone and Setup
```bash
git clone <repository-url>
cd veritas
cp backend/.env.example backend/.env
# Edit backend/.env with your configuration
```

### 2. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### 3. Manual Development Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Service Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **SearxNG**: http://localhost:8888
- **ChromaDB**: http://localhost:8002

## 📡 API Endpoints

### Verification Endpoints
- `POST /api/v1/verify-post` - Submit post for verification
- `GET /api/v1/verification-status/{verification_id}` - Get verification status

### Reputation Endpoints
- `GET /api/v1/user-reputation/{nickname}` - Get user reputation
- `GET /api/v1/reputation-stats` - Get platform statistics
- `GET /api/v1/users-with-warnings` - Get users with warnings
- `GET /api/v1/leaderboard` - Get reputation leaderboard

### System Endpoints
- `GET /` - Service information
- `GET /health` - Health check
- `WS /ws` - WebSocket for real-time updates

## 🔄 WebSocket Events

### Client → Server
- `ping` - Heartbeat
- `subscribe` - Subscribe to verification updates

### Server → Client
- `session_established` - Session ID assignment
- `verification_progress` - Real-time progress updates
- `verification_complete` - Final results
- `verification_error` - Error notifications

## 📊 Reputation System

### Scoring Algorithm
- **True**: +1.0 points
- **Partially True**: +0.5 points
- **False**: -2.0 points
- **Ironic**: -1.0 points

### Warning System
- **Notification**: Score ≤ -10.0
- **Warning**: Score ≤ -20.0
- **Tracking**: All user interactions logged

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

## 🔧 Development

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Code Quality
```bash
# Backend
cd backend
black app/ agent/
flake8 app/ agent/
mypy app/ agent/

# Frontend
cd frontend
npm run lint
npm run format
```

## 🐳 Docker Services

### Production Deployment
```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale backend=3
```

### Service Health Checks
All services include comprehensive health checks:
- **PostgreSQL**: Connection and query tests
- **Redis**: Ping response
- **SearxNG**: HTTP endpoint availability
- **ChromaDB**: Port connectivity
- **Backend**: API health endpoint

## 📁 Project Structure

```
veritas/
├── backend/                     # FastAPI backend
│   ├── app/                    # Main application
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Application services
│   │   ├── config.py          # Configuration management
│   │   ├── database.py        # Database models and setup
│   │   ├── schemas.py         # Pydantic schemas
│   │   └── main.py           # FastAPI application
│   ├── agent/                 # AI agent system
│   │   ├── analyzers/        # Specialized analyzers
│   │   ├── fact_checkers/    # Domain-specific fact checkers
│   │   ├── services/         # Core verification services
│   │   ├── pipeline/         # Verification pipeline
│   │   ├── models/           # Data models
│   │   └── workflow_coordinator.py
│   ├── alembic/              # Database migrations
│   └── tests/                # Test suites
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # Frontend services
│   │   └── utils/           # Utility functions
│   └── public/              # Static assets
├── docker/                   # Docker configurations
├── data/                     # Data and migrations
└── docs/                     # Project documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for advanced LLM orchestration
- **Ollama** for local multimodal AI hosting
- **SearxNG** for privacy-focused search
- **ChromaDB** for vector database capabilities
- **FastAPI** for high-performance API framework
- **React** for modern frontend development

## 📞 Support

For questions, issues, or contributions:
- Open an issue in the repository
- Use the discussions area for general questions
- Check the [documentation](docs/) for detailed information

---

**Veritas** - *Where Truth Meets Technology* 🔍✨


Отличная задача! Вы совершенно правы, что принципы улучшения должны быть одинаковыми, и мы можем применить тот же строгий подход к промпту для анализа изображений.

Я проанализировал ваш текущий промпт, код и Pydantic-модели, а также провел поиск по лучшим практикам для мультимодальных систем (Vision-Language Models). Давайте действовать по нашему проверенному плану.

### Результаты Исследования: Лучшие практики для Мультимодальных Промптов

1.  **Явное разделение задач (Explicit Task Decomposition):** Для сложных задач, включающих и распознавание текста (OCR), и анализ визуальных элементов, и структурирование данных, критически важно разбить процесс на шаги. Модели работают лучше, когда им говорят: "Сначала сделай А, потом сделай Б, а потом на основе А и Б сделай В".
2.  **Пошаговое мышление (Chain-of-Thought - CoT):** Это развитие предыдущего пункта. Промпт должен не просто давать список правил, а вести модель по цепочке рассуждений. Например: "Шаг 1: Опиши, что видишь. Шаг 2: Извлеки весь текст. Шаг 3: Проанализируй связь между текстом и визуальными элементами. Шаг 4: Сформируй JSON".
3.  **Предоставление Примеров (Few-Shot Prompting):** Для сложных структурированных выводов, как ваш JSON, предоставление одного или двух высококачественных примеров (`<example>...</example>`) в промпте работает на порядок лучше, чем простое описание формата. Модель "видит" эталон и следует ему.
4.  **Четкое указание на источник информации:** Промпт должен заставлять модель четко разделять информацию, извлеченную из текста на картинке, от выводов, сделанных на основе визуальных элементов.
5.  **Прямые инструкции по обработке ошибок:** Вместо того чтобы исправлять ошибки форматирования (как `_clean_json_output`), лучше добавить в промпт явные инструкции, которые предотвратят их появление. Прямое указание "Do not use underscores in numbers" — хороший пример, но его можно усилить.

---

### Анализ Вашего Текущего Промпта

*   **Сильные стороны:** Он уже использует роль, указывает на JSON, передает `{format_instructions}` и текущую дату. Это отличная база.
*   **Точки роста:**
    *   Он представляет собой набор правил, а не пошаговый процесс.
    *   Он смешивает задачи (извлечение, анализ, структурирование) в одном потоке инструкций.
    *   Он полагается на внешний код (`_clean_json_output`) для исправления ошибок, которые можно предотвратить на уровне промпта.

---

### Финальный Промпт 2.0 (для анализа изображений)

Давайте применим лучшие практики и создадим новый, более мощный и надежный промпт.

**System:**
You are a meticulous AI analyst specializing in social media content verification. Your task is to analyze an image of a social media post by following a structured, step-by-step process and extracting key information into a precise JSON format.

**Current Date for Context:** `{current_date}`

**Your Step-by-Step Analysis Process:**

**Step 1: Raw Data Extraction**
*   **Transcribe Text:** Perform OCR to transcribe ALL visible text from the image verbatim. Pay close attention to usernames, timestamps (e.g., "15h ago", "May 21"), years, and the main body of the post.
*   **Describe Visuals:** Describe any significant non-text visual elements (e.g., "A bar chart showing rising values," "A profile picture of a cat," "UI elements like like/retweet buttons").

**Step 2: Analysis and Structuring**
*   Based on the raw data from Step 1, analyze and structure the information.
*   Determine the overarching claim (Primary Thesis) and break it down into smaller, verifiable supporting facts.
*   Identify the post's topic and assess for irony.

**Step 3: JSON Assembly**
*   Construct the final JSON object based on your analysis.
*   **Crucial Formatting Rules:**
    *   The JSON object MUST be perfectly valid.
    *   All numbers must be standard JSON numbers (e.g., `4970`, `150.5`). **DO NOT** use underscores, commas, or any other formatting within numbers.
    *   Follow the provided Pydantic format instructions precisely.

**High-Quality Example of Final Output:**

```json
<example>
{
  "username": "@CryptoNewsDaily",
  "post_date": "21h ago",
  "mentioned_dates": [
    "May 21"
  ],
  "extracted_text": "BREAKING: BlackRock just snapped up another 4,970 BTC on May 21, adding to their massive crypto portfolio. #Bitcoin #BlackRock",
  "fact_hierarchy": {
    "primary_thesis": "On May 21, BlackRock made a significant Bitcoin purchase of 4,970 BTC.",
    "supporting_facts": [
      {
        "description": "BlackRock purchased Bitcoin.",
        "context": {
          "entity": "BlackRock",
          "action": "purchase",
          "asset": "Bitcoin"
        }
      },
      {
        "description": "The purchase amount was 4,970 BTC.",
        "context": {
          "amount": 4970,
          "asset": "BTC"
        }
      },
      {
        "description": "The purchase occurred on May 21.",
        "context": {
          "date": "May 21"
        }
      }
    ]
  },
  "primary_topic": "financial",
  "irony_assessment": "not_ironic",
  "visual_elements_summary": "The image contains text on a dark background with a crypto-themed logo."
}
</example>
```

{format_instructions}

**Human:**
Please analyze this social media post image and provide the structured JSON output.

---

### Почему эта версия значительно лучше:

1.  **Пошаговый процесс (CoT):** Промпт заставляет модель сначала собрать сырые данные (текст, визуал), потом их проанализировать, и только потом собрать JSON. Это снижает когнитивную нагрузку и уменьшает количество ошибок.
2.  **Предотвращение ошибок:** Вместо того чтобы чистить JSON после, мы даем модели сверхчеткие инструкции и пример правильного формата чисел (`4970`). Это должно сделать ваш метод `_clean_json_output` ненужным.
3.  **Высококачественный пример (Few-Shot):** Блок `<example>` — это самый мощный инструмент здесь. Он не просто описывает структуру, а **показывает** ее на реалистичном примере, включая вложенные объекты. Модель будет стремиться воспроизвести этот паттерн.
4.  **Ясность и логика:** Промпт теперь не просто набор правил, а логический рабочий процесс. Это делает его более понятным для LLM и, как следствие, более надежным в исполнении.
5.  **Повышение точности:** Требование сначала извлечь весь текст дословно (`extracted_text`), а потом уже формировать тезис, заставляет модель основывать свои выводы на реальных данных из изображения, а не на своих предположениях.

Этот новый промпт должен значительно повысить качество и надежность работы `ImageAnalysisService` и минимизировать сбои при парсинге.
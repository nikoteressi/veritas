Понял! Отлично, "Veritas" звучит гораздо лучше и соответствует цели проекта.

Вот обновленные документы с новым названием проекта:

---

## Veritas: AI-Powered Social Post Verifier

### Technical Description

**Project Concept:**
Veritas is an advanced AI-driven agent system designed for the automated verification of facts within social media posts. The system accepts a screenshot of a social media post and a text prompt from the user. It leverages a multimodal Large Language Model (Llama 4) for deep comprehension of the image content (including text, graphics, and visual context) and performs intelligent fact-checking. A core innovation is the dynamic deployment of specialized AI agents tailored to specific subject domains (e.g., finance, medicine, science) to ensure highly accurate verification. Furthermore, Veritas maintains a detailed user reputation system, providing real-time feedback and issuing warnings based on the veracity of shared content.

**Key Functional Capabilities:**

1.  **Multimodal Input Analysis:** Processes screenshots using a multimodal Llama 4 model to accurately extract text, analyze charts and graphs, identify user nicknames, and understand the overall visual context, including attempting to detect irony or sarcasm.
2.  **Intelligent Fact Verification (RAG):** Utilizes Llama 4 as the central intelligence engine, augmented by access to up-to-date external information sources via a self-hosted web search API.
3.  **Dynamic Agent Specialization:** Automatically classifies the post's subject matter (e.g., financial, political, medical, scientific) and activates specialized AI agents equipped with relevant tools and knowledge bases for more precise fact-checking.
4.  **Detailed User Reputation System:** Maintains comprehensive statistics for each user, categorizing verified posts as truthful, partially truthful, false, or ironic. The system issues automated warnings and notifications upon reaching thresholds of misinformation shared.
5.  **Real-time Feedback:** Provides users with live updates on the fact-checking process, enhancing transparency and user experience.
6.  **Robust & Scalable Backend:** Built with an asynchronous architecture to efficiently handle concurrent requests and scale as needed.

---

### Technology Stack

### I. Backend (Python)

*   **API Framework:**
    *   **FastAPI:** High-performance, asynchronous web framework for building RESTful APIs, supporting WebSockets for streaming updates, and efficient handling of file uploads (screenshots).
    *   **Uvicorn:** ASGI server for running FastAPI applications.
*   **AI/LLM Orchestration & Multimodality:**
    *   **LangChain:** Comprehensive framework for developing LLM-powered applications. Used for:
        *   Agent orchestration and execution flow.
        *   Integration with Ollama (Llama 4).
        *   Prompt management and templating.
        *   Defining and utilizing external tools (web search, databases).
        *   Implementing the Retrieval Augmented Generation (RAG) pipeline.
    *   **Ollama:** For local hosting and running of a **multimodal Llama 4** model (or equivalent VLM) on your server, enabling direct image understanding.
*   **External Web Search (for RAG):**
    *   **SearxNG:** Self-hosted, open-source metasearch engine. Configured with JSON API output for programmatic access and integrated with LangChain. Chosen for full control over search requests, avoiding CAPTCHAs and rate limits of public search APIs. Requires dedicated server deployment and maintenance.
*   **Databases:**
    *   **PostgreSQL:** Robust, scalable, and reliable relational database management system for storing user profiles and their detailed reputation metrics.
    *   **SQLAlchemy:** Python Object-Relational Mapper (ORM) for intuitive and secure interaction with PostgreSQL.
    *   **ChromaDB:** Lightweight vector database. Utilized for:
        *   Caching results of frequently checked facts.
        *   Storing embeddings for custom knowledge bases (if further expanded).
*   **Other Libraries:**
    *   `python-dotenv`: For managing environment variables (e.g., API keys, database credentials).
    *   `requests`: For general-purpose HTTP requests to external services where native LangChain integrations might not exist or be preferred.
    *   `psycopg2-binary`: PostgreSQL adapter for Python.

### II. Frontend (Web)

*   **Framework:**
    *   **React:** A declarative, component-based JavaScript library for building interactive user interfaces. Known for its large ecosystem and strong community support.
*   **State Management:**
    *   **React Context API / Zustand / Jotai:** For managing global application state, such as loading statuses, user data, and request history. Choice depends on the complexity of state requirements.
*   **Styling:**
    *   **Tailwind CSS:** A utility-first CSS framework for rapidly building custom user interfaces with highly configurable design.
*   **HTTP Client:**
    *   **Axios / Fetch API:** For sending HTTP requests to the FastAPI backend, including uploading images and submitting text prompts.
*   **Real-time Communication:**
    *   **WebSockets API (native browser):** To establish and manage real-time, bidirectional communication with the FastAPI backend, receiving streamed updates on the fact-checking process.
*   **Component Libraries (Optional, for accelerated development):**
    *   **Ant Design / Material-UI / Chakra UI:** Pre-built UI component libraries to expedite prototyping and enhance the aesthetic appeal of the interface.
*   **File Upload:**
    *   `react-dropzone`: A user-friendly library for implementing drag-and-drop file upload functionality.

---

### Architectural Flow:

1.  **User Interface (React):** A user uploads a screenshot and inputs a text prompt via the web application.
2.  **API Request (Axios/Fetch):** The React application sends a `multipart/form-data` request (containing the screenshot image and text prompt) to the FastAPI backend.
3.  **Backend Processing Initiation (FastAPI):**
    *   FastAPI receives the image and prompt.
    *   It initiates a WebSocket session to provide real-time feedback to the frontend.
    *   The request is then passed to the LangChain agent for processing.
4.  **Agent Orchestration (LangChain):**
    *   **Multimodal Analysis:** LangChain invokes Ollama with the multimodal Llama 4 model, passing both the image and prompt. Llama 4 extracts text, identifies the user's nickname, analyzes any visual data (graphs), determines the post's primary topic, and attempts to detect nuances like irony.
    *   **User Reputation Check:** The agent queries PostgreSQL (via SQLAlchemy) to retrieve the user's current reputation metrics for the extracted nickname.
    *   **Topic Classification:** Llama 4 classifies the post's subject matter (e.g., financial, medical, scientific).
    *   **Dynamic Tool Selection:** Based on the topic classification, LangChain dynamically activates and routes queries to specialized "tools" (e.g., specific queries to the self-hosted SearxNG instance for financial data).
    *   **Web Search & Verification:** Llama 4 utilizes the search results from SearxNG to verify claims made in the post.
    *   **Verdict & Reputation Update:** Llama 4 generates a verdict (truthful, partially truthful, false, ironic) and provides a justification. LangChain updates the corresponding user counters in PostgreSQL.
    *   **Response Generation:** The agent compiles the final response for the user, including the verdict, its rationale, and relevant reputation information.
5.  **Streaming Updates (WebSockets):** FastAPI sends intermediate status updates and, finally, the conclusive verification result back to the React frontend via the WebSocket connection.
6.  **Results Display (React):** The React application dynamically renders the received data to the user, showcasing the verification outcome and any reputation alerts.

---
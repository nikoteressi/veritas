## Project Implementation Plan: Veritas

**Target Audience:** Coding Agent (Claude Sonnet 4)

**Objective:** Develop a full-stack, AI-powered Fact-Checking Agent as described in the previous "Technical Description" and "Technology Stack".

**Key Assumption:** Ollama server (running multimodal Llama 4) is **already deployed on a remote server** and accessible via a network address. No local Ollama setup required.

---

### Phase 1: Environment & Infrastructure Setup

1.  **Project Structure:**
    *   Create a root directory: `Veritas`
    *   Inside, create: `backend/`, `frontend/`, `scripts/`, `data/`, `docker/`
    *   Set up a Python virtual environment in `backend/`.
    *   Initialize a Node.js project in `frontend/`.

2.  **Database Setup (PostgreSQL):**
    *   **Action:** Provision a PostgreSQL instance (e.g., via Docker, cloud service, or local installation if not already available).
    *   **Configuration:** Create a database `veritas_db` and a user `veritas_user` with appropriate permissions.
    *   **Environment Variables:** Add PostgreSQL connection details (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) to `backend/.env`.

3.  **Vector Database Setup (ChromaDB):**
    *   **Action:** Install ChromaDB within the Python environment. For initial simplicity, assume embedded mode for development. If persistence or sharing across processes is required, consider running ChromaDB in client/server mode (e.g., via Docker).

4.  **SearxNG Setup (Self-hosted Web Search):**
    *   **Action:** Deploy a SearxNG instance (recommended via Docker Compose).
    *   **Configuration:**
        *   Locate `settings.yml` (e.g., in `/etc/searxng/settings.yml` or the mounted volume if Docker).
        *   **Enable JSON output:** Find `search: formats:` and ensure `- json` is present alongside `- html`.
        *   **Disable quotas (if necessary for high volume):** Find `server: quotas:` and set `enabled: false` if it exists, or remove/comment out related lines.
        *   **Environment Variables:** Add `SEARXNG_URL` (e.g., `http://localhost:8888` or your server IP/hostname) to `backend/.env`.
    *   **Verification:** Test by accessing `SEARXNG_URL/search?q=test&format=json` in a browser or via `curl`.

5.  **Ollama Connection Configuration:**
    *   **Environment Variables:** Add `OLLAMA_BASE_URL` (e.g., `http://192.168.11.130:11434`) to `backend/.env`.

### Phase 2: Backend Development (FastAPI & LangChain)

1.  **FastAPI Core Application:**
    *   **Dependencies:** Install `fastapi`, `uvicorn`, `python-dotenv`, `psycopg2-binary`, `sqlalchemy`, `langchain`, `langchain-community`, `chromadb`, `Pillow` (for image loading/pre-processing before base64 encoding).
    *   **`backend/main.py`:**
        *   Initialize FastAPI app.
        *   Load environment variables.
        *   Define `POST /verify_post` endpoint:
            *   Accepts `UploadFile` (screenshot) and `prompt` (string).
            *   Implement `StreamingResponse` for real-time updates via WebSockets.
        *   Implement `GET /user_reputation/{nickname}` endpoint.
    *   **`backend/database.py`:**
        *   SQLAlchemy setup: `Base`, `engine`, `SessionLocal`.
        *   Define `User` model with relevant fields: `id`, `nickname`, `false_count`, `partially_true_count`, `ironic_count`, `true_count`, `total_posts_checked`, `last_checked_date`.
        *   Functions for CRUD operations on `User` table (e.g., `get_user`, `create_user_if_not_exists`, `update_user_counts`).
    *   **`backend/vector_db.py`:**
        *   Initialize ChromaDB client.
        *   Functions for adding/retrieving embeddings (e.g., for caching verified facts or context).

2.  **LangChain Agent Implementation (`backend/agent/`):**

    *   **`backend/agent/llm.py`:**
        *   Initialize Ollama LLM (`ChatOllama`) using `OLLAMA_BASE_URL`.
        *   Specify the multimodal Llama 4 model name (e.g., `llava:latest`, `llama4:multimodal`).
        *   Helper function to encode PIL Image objects to base64 string for Llama 4 input.

    *   **`backend/agent/tools.py`:**
        *   **`SearxNGSearchTool`:** Implement LangChain tool using `SearxSearchWrapper` initialized with `SEARXNG_URL`.
            *   Define a clear `name` and `description` for the tool that Llama 4 can understand.
        *   **`PostgresDBTool`:** Implement a custom LangChain tool for database interactions.
            *   Methods: `update_user_reputation_counts(nickname, verdict_category)`, `get_user_reputation(nickname)`.
            *   Define precise `name` and `description` for each method as a tool.

    *   **`backend/agent/prompts.py`:**
        *   Define various `ChatPromptTemplate`s and `SystemMessagePromptTemplate`s for different stages:
            *   **Initial Analysis Prompt (Multimodal):** Instructs Llama 4 to:
                *   Describe the image content (text, graphics, layout, visual cues).
                *   Extract all visible text verbatim.
                *   Identify the user's nickname, explicitly stating its typical location in social media.
                *   List all distinct claims/statements to be fact-checked.
                *   Classify the post's primary topic (e.g., `finance`, `medical`, `political`, `scientific`, `general`, `humorous/ironic`).
                *   Provide an initial assessment of potential irony or sarcasm.
            *   **Fact-Checking Prompt (General):** Generic instructions for the agent to verify claims using available tools.
            *   **Fact-Checking Prompt (Specialized):** Dynamic template to inject domain-specific instructions based on topic classification (e.g., "You are a seasoned financial analyst. Verify these claims rigorously using your financial knowledge and tools.").
            *   **Verdict Generation Prompt:** To synthesize search results and original claims into a clear, concise verdict (True, Partially True, False, Ironic) and a brief justification, citing sources where possible.
            *   **Reputation Update Prompt:** To guide Llama 4 on how to categorize the post's verdict for user reputation.

    *   **`backend/agent/core.py`:**
        *   **Main Agent Logic:**
            *   Create an `AgentExecutor` with the multimodal Llama 4 as the LLM.
            *   Define the core `tools` available to the agent (SearxNGSearchTool, PostgresDBTool).
            *   **Step 1: Multimodal Understanding:**
                *   Receive image (as bytes) and prompt.
                *   Convert image to base64.
                *   Call Llama 4 with the `Initial Analysis Prompt` and base64-encoded image.
                *   Parse Llama 4's structured output: extracted text, nickname, identified claims, primary topic, initial irony assessment.
                *   Yield status: "Image analyzed. Extracting claims..."
            *   **Step 2: User Reputation Retrieval:**
                *   Use `PostgresDBTool` to retrieve the user's current `false_count`, `true_count`, `partially_true_count`, `ironic_count`.
                *   Yield status: "Checking user's reputation..."
            *   **Step 3: Dynamic Fact-Checking Pipeline:**
                *   Based on the classified topic from Step 1, dynamically adjust the `Fact-Checking Prompt` (e.g., add role-playing for "medical expert").
                *   Iterate through each identified claim:
                    *   Use Llama 4 and `SearxNGSearchTool` to perform targeted searches for the claim.
                    *   Llama 4 analyzes search results to determine the veracity of that specific claim.
                    *   Yield status: "Verifying claim X of Y..."
            *   **Step 4: Overall Verdict & Justification:**
                *   Llama 4 synthesizes individual claim verifications and original context into an overall verdict for the entire post and provides a brief explanation.
                *   Refine irony detection if the initial assessment was ambiguous, possibly with a follow-up LLM call.
                *   Yield status: "Finalizing verdict..."
            *   **Step 5: Update User Reputation:**
                *   Map the final verdict to the appropriate reputation category (`true`, `false`, `partially_true`, `ironic`).
                *   Use `PostgresDBTool` to update the user's `false_count`, `true_count`, `partially_true_count`, `ironic_count`, and `total_posts_checked`.
                *   Implement the logic for warnings (e.g., after 3 false posts) and notifications (e.g., after 5 false posts), to be included in the final response.
                *   Yield status: "Updating user reputation..."
            *   **Step 6: Generate Final Response:**
                *   Format the final verdict, detailed justification, and the updated reputation status with any warnings/notifications.
                *   Yield final result.

### Phase 3: Frontend Development (React)

1.  **Frontend Project Setup:**
    *   **Dependencies:** Install `react`, `react-dom`, `axios`, `tailwindcss`, `react-dropzone`.
    *   **`frontend/src/index.js` / `App.js`:**
        *   Basic React app structure.
        *   Tailwind CSS setup.

2.  **Components:**
    *   **`UploadForm.js`:**
        *   File input (using `react-dropzone`) for screenshot. Display preview of selected image.
        *   Textarea for user prompt.
        *   Submit button.
        *   State management for selected file, prompt text, and submission status.
    *   **`VerificationResults.js`:**
        *   Displays the real-time streamed updates received via WebSocket.
        *   Clearly presents the final verdict (True/False/Partially True/Ironic), its justification, and the user's reputation information.
        *   Conditionally renders warning/notification messages based on reputation.
    *   **`UserReputationDisplay.js` (Optional):**
        *   A separate component (potentially on a different route or a modal) to search for a user by nickname and display their full reputation history (`true_count`, `false_count`, etc.).

3.  **API Integration:**
    *   **`frontend/src/api.js`:**
        *   `uploadPostForVerification(file, prompt)` function: Uses Axios to send `multipart/form-data` to the `POST /verify_post` endpoint.
        *   **WebSocket Handling:** Establish a WebSocket connection to the FastAPI endpoint (e.g., `ws://your-backend-ip:8000/ws`).
            *   Implement logic to listen for messages, parse them (e.g., JSON), and update the UI state to display real-time progress.
        *   `getUserReputation(nickname)` function: Uses Axios to fetch data from `GET /user_reputation/{nickname}`.

4.  **User Experience (UX):**
    *   Implement clear loading indicators and progress messages during the verification process.
    *   Ensure responsive design for various screen sizes.
    *   Provide clear error messages for failed requests or processing issues.

### Phase 4: Deployment & Testing

1.  **Backend Deployment:**
    *   **Action:** Containerize FastAPI application using Docker.
    *   **`backend/Dockerfile`:** Create a Dockerfile to build your FastAPI application image.
    *   **`docker-compose.yml` (Recommended):** Orchestrate FastAPI, PostgreSQL, and SearxNG services for simplified deployment. Ensure network configuration allows communication between these services and to the remote Ollama server.
    *   **Deployment:** Deploy the Docker containers to your chosen server environment.

2.  **Frontend Deployment:**
    *   **Action:** Build the React application for production (`npm run build`).
    *   **Deployment:** Serve the static files (e.g., via Nginx, Caddy, or a static site hosting service). Ensure the frontend can connect to your deployed FastAPI backend (update `AXIOS_BASE_URL` in frontend).

3.  **Testing:**
    *   **Unit Tests:** Develop unit tests for critical components (e.g., database operations, image base64 encoding, specific LangChain tool functions).
    *   **Integration Tests:** Test the flow between FastAPI and LangChain, and between LangChain and external tools (SearxNG, PostgreSQL).
    *   **End-to-End Tests:** Verify the complete user journey from screenshot upload to final verdict display, including real-time updates.
    *   **Edge Cases:** Rigorously test with complex images (multiple texts, dense graphics), highly nuanced/sarcastic content, clearly false claims, partially true statements, and users at various reputation levels (new, warning, notified).
    *   **Performance Testing:** Monitor response times and resource utilization under load.

### Post-Implementation Considerations:

*   **Observability:** Implement robust logging (e.g., Python's `logging` module, `winston` for Node.js) and monitoring (e.g., Prometheus/Grafana) for both backend and frontend applications.
*   **Security:** Implement best practices for API key management, input validation, rate limiting, and secure WebSocket connections (WSS).
*   **Performance Optimization:** Continuously optimize image processing (before LLM), LLM inference time (considering hardware/model quantization), and database query efficiency.
*   **Scalability:** Plan for future scaling by considering load balancers, potential database sharding, and dedicated LLM serving infrastructure if user traffic grows significantly.
*   **Cost Management:** Monitor usage of any potential paid APIs (though SearxNG is self-hosted) and server resource consumption.
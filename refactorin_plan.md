# Veritas Project Refactoring Plan

This document outlines a series of recommendations to refactor and enhance the Veritas project. The current architecture is solid, with a good separation of concerns. These suggestions aim to improve maintainability, scalability, and developer experience by leveraging best practices in Python, React, and general software architecture.

## 1. Backend (Python/FastAPI)

The backend is well-structured, but we can improve configuration, streamline the core business logic, and optimize asynchronous operations.

### 1.1. Unify Configuration Management

* **Current State:** Configuration is split between `app/config.py` (using `pydantic-settings`) and a custom `agent/services/configuration_service.py`.
* **Recommendation:** Consolidate all configuration into the `app/config.py` module.
* **Action Items:**
  1. Migrate all settings from `ConfigurationService` into the `pydantic_settings.BaseSettings` class in `app/config.py`.
  2. Remove `agent/services/configuration_service.py`.
  3. Update all modules to import settings from a single source (`app.config`).
  * **Benefit:** Creates a single source of truth for configuration, simplifying management and reducing ambiguity.

### 1.2. Refactor the Verification Pipeline

* **Current State:** The `VerificationPipeline` class in `agent/services/verification_pipeline.py` likely hardcodes the sequence of analysis steps.
* **Recommendation:** Make the pipeline more declarative and data-driven.
* **Action Items:**
  1. **Create a Context Model:** Define a Pydantic model, e.g., `VerificationContext`, to hold all data passed between steps (input text, image URLs, intermediate analysis results, etc.).
  2. **Isolate Pipeline Steps:** Refactor each analyzer (e.g., `temporal_analysis.py`, `motives_analyzer.py`) into a standalone class or function that accepts and returns the `VerificationContext` object.
  3. **Dynamic Pipeline Construction:** Modify `VerificationPipeline` to dynamically build its sequence of steps from a configuration list, rather than having it hardcoded.
  * **Benefit:** Improves modularity, simplifies testing of individual steps, and allows for easy reordering or disabling of analyzers without code changes.

### 1.3. Improve Asynchronous Initialization

* **Current State:** The `AgentManager` in `agent/services/agent_manager.py` performs potentially blocking I/O operations in its `__init__` method.
* **Recommendation:** Use an async factory pattern for initialization.
* **Action Items:**
  1. Create an `async def create_agent_manager(...)` function.
  2. Move the initialization logic from `AgentManager`'s constructor into this factory function.
  3. Use FastAPI's `lifespan` event handler in `app/main.py` to call this factory once on application startup and attach the manager instance to the app state.
  * **Benefit:** Ensures the application starts in a fully non-blocking, asynchronous manner, which is crucial for performance and stability.

### 1.4. Optimize Database Interactions

* **Current State:** The `app/crud.py` and `app/routers/reputation.py` modules contain functional but improvable database queries.
* **Recommendation:** Leverage modern SQLAlchemy features for more efficient and concise queries.
* **Action Items:**
  1. In `UserCRUD.get_or_create_user`, consider using a database-level `INSERT ... ON CONFLICT DO NOTHING` (if using PostgreSQL) to handle race conditions more efficiently than a separate `SELECT` and `INSERT`.
  2. In `reputation.py`, refactor multiple separate queries for statistics into a single, more complex query using SQLAlchemy's aggregate functions (`func.sum`, `func.count`, `func.avg`).
  * **Benefit:** Reduces the number of database round-trips, significantly improving performance under load.

## 2. Frontend (React)

The frontend uses modern React practices. The focus here is on improving state management and component structure.

### 2.1. Centralize State Management Logic

* **Current State:** State logic is managed across several custom hooks (`useWebSocketService`, `useVerificationState`) within `App.jsx`.
* **Recommendation:** Encapsulate the related state and logic into a single, higher-level custom hook.
* **Action Items:**
  1. Create a `useVeritas()` hook.
  2. This hook will internally use `useWebSocketService` and manage the verification state (`isLoading`, `result`, `error`).
  3. It should expose a clean interface to the `App` component, e.g., `{ status, result, error, submitVerification }`.
  4. Refactor `App.jsx` to use this single hook.
  * **Benefit:** Decouples the main `App` component from the complexities of state management, making it cleaner and easier to understand.

### 2.2. Decompose Large Components

* **Current State:** `VerificationResults.jsx` contains conditional logic to render multiple states (loading, initial, error, results).
* **Recommendation:** Split the component based on the state it represents.
* **Action Items:**
  1. Create the following specialized components:
     * `LoadingState.jsx`: Displays the loading animation/progress.
     * `ResultDisplay.jsx`: Renders the formatted verification results.
     * `InitialState.jsx`: Shows the initial welcome message and instructions.
     * `ErrorState.jsx`: Displays a user-friendly error message.
  2. Use the parent component (`VerificationResults.jsx`) to simply orchestrate which of these smaller components to render based on the current state.
  * **Benefit:** Enhances readability, simplifies maintenance, and promotes component reuse.

## 3. General Architecture & Best Practices

### 3.1. Adopt a Modern Dependency Manager

* **Current State:** Dependencies are listed in `backend/requirements.txt`.
* **Recommendation:** Switch to `Poetry` or `pip-tools`.
* **Action Items:**
  1. Initialize a `pyproject.toml` file using `poetry init`.
  2. Add all dependencies to this file.
  3. Generate a `poetry.lock` file to ensure deterministic builds.
  4. Update the project's setup and CI/CD instructions to use Poetry.
  * **Benefit:** Provides robust dependency resolution, prevents conflicts, and ensures reproducible environments for all developers.

### 3.2. Implement a Comprehensive Testing Suite

* **Current State:** The `tests/` directories are empty.
* **Recommendation:** Build out a test suite covering all layers of the application.
* **Action Items:**
  1. **Unit Tests (Pytest):** Write tests for individual functions and classes, especially the analyzers in `agent/` and helper functions.
  2. **Integration Tests (Pytest):** Write tests for the `VerificationPipeline` to ensure analyzers work correctly together. Test the FastAPI endpoints to verify request/response contracts.
  3. **End-to-End (E2E) Tests:** Use a framework like `Playwright` or `Cypress` to simulate user flows from the frontend (uploading a file) to the backend and back.
  * **Benefit:** Increases code quality, prevents regressions, and provides confidence when shipping new features.

### 3.3. Enhance Docker for Production

* **Current State:** A single `docker-compose.yml` is used for development.
* **Recommendation:** Create a separate, optimized configuration for production.
* **Action Items:**
  1. Create a `docker-compose.prod.yml` that overrides the development configuration.
  2. In the production service for the backend, switch the `command` from `uvicorn` to a production-grade ASGI server like `gunicorn -w 4 -k uvicorn.workers.UvicornWorker`.
  3. Ensure debugging and auto-reload features are disabled in the production environment.
  4. Verify that all secrets are passed securely via environment variables and are not hardcoded.
  * **Benefit:** Creates a robust, secure, and performant setup for deployment.
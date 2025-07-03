## **Refactoring Plan**

### **1. Summary of Findings**

The codebase has a strong asynchronous foundation and a logical high-level structure. However, it is compromised by significant architectural flaws and implementation bugs that impact scalability, maintainability, and correctness.

*   **Strengths**:
    *   Fully asynchronous design using FastAPI.
    *   Clear project structure (`agent`, `app`, `routers`).
    *   Use of Pydantic for schema validation.

*   **Weaknesses**:
    *   **"God Class" Anti-pattern**: `VeritasAgent` in `agent/core.py` is a monolithic class with far too many responsibilities.
    *   **Volatile State Management**: The `ConnectionManager` uses an in-memory dictionary for session state, making the application unscalable and subject to data loss on restart.
    *   **Dead/Duplicate Code**: The `agent/core.py` file is duplicated, and the `image_processing.py` file is 99% unused.
    *   **Critical Bugs**: The application has an incorrect implementation of database sessions in background tasks and a non-functional API endpoint for retrieving results.
    *   **Untyped Data Flow**: Core components communicate using unstructured dictionaries, which is error-prone.

### **2. Identified Issues & Refactoring Recommendations**

#### **Issue 1: Monolithic Agent (`VeritasAgent`)**
*   **Context**: `agent/core.py` contains the `VeritasAgent` class, which handles orchestration, LLM interaction, database operations, and more. This violates the Single Responsibility Principle (SRP).
*   **Recommendation**: Decompose `VeritasAgent` into smaller, service-oriented classes.
    1.  **`VerificationOrchestrator`**: Manages the high-level verification workflow, calling other services.
    2.  **`ImageAnalysisService`**: Handles the `_analyze_image` logic.
    3.  **`FactCheckingService`**: Manages `_fact_check_claims` and tool interactions.
    4.  **`VerdictService`**: Generates the final verdict (`_generate_verdict`).
    5.  **`ReputationService`**: Encapsulates all user reputation logic and database interactions.
*   **Design Pattern**: Apply the **Single Responsibility Principle** to create a more modular, service-oriented architecture.

#### **Issue 2: Unstructured Data Flow**
*   **Context**: Large, untyped dictionaries are passed between methods, making data contracts implicit and fragile.
*   **Recommendation**: Define explicit data transfer objects (DTOs) using Pydantic's `BaseModel`.
    *   Create `FactCheckResult(BaseModel)`
    *   Create `VerdictResult(BaseModel)`
    *   Continue using the existing `ImageAnalysisResult(BaseModel)`.
*   **Reasoning**: This creates clear, validated, and self-documenting data contracts between services.

#### **Issue 3: Obsolete and Duplicate Code**
*   **Context**: `agent/core.py` is a duplicate of `backend/agent/core.py`. `backend/app/image_processing.py` is almost entirely unused.
*   **Recommendation**:
    1.  Delete the duplicate file: `agent/core.py`.
    2.  Relocate the one used function, `validate_image`, from `image_processing.py` to `validators.py`.
    3.  Delete the now-empty `image_processing.py` file.
*   **Reasoning**: Eliminating dead and duplicated code is crucial for reducing complexity and avoiding confusion.

#### **Issue 4: In-Memory State Management**
*   **Context**: `websocket_manager.py` stores session progress in a local dictionary, which is a critical scalability and reliability bottleneck.
*   **Recommendation**: Replace the in-memory dictionary with **Redis**. It is a fast, persistent, and scalable solution for managing session state and background job data. The `verification_sessions` dictionary should be refactored to use Redis hashes.
*   **Reasoning**: This change is essential to enable horizontal scaling and prevent data loss.

#### **Issue 5: Critical Router Bugs**
*   **Context**:
    1. A database session is passed to a background task in `routers/verification.py`, which will fail.
    2. The `/verification-status/{verification_id}` endpoint is not implemented.
*   **Recommendation**:
    1.  Modify `run_verification_with_websocket` to create and manage its own database session.
    2.  Implement the `/verification-status/{verification_id}` endpoint to query the database and return the result for a given ID.
*   **Reasoning**: These are critical bug fixes required for the application to function correctly and reliably.

### **3. Prioritized Refactoring Plan**

The plan is prioritized to fix the most critical issues first.

**Phase 1: Critical Bug Fixes & Cleanup (Highest Priority)**

1.  **Fix DB Session Bug**: Refactor `run_verification_with_websocket` in `routers/verification.py` to manage its own database session lifecycle.
2.  **Delete Duplicate File**: Delete `agent/core.py`.
3.  **Remove Obsolete Code**: Move `validate_image` to `validators.py` and delete `image_processing.py`. Update the import in `routers/verification.py`.

**Phase 2: Architectural Foundation**

4.  **Integrate Redis**: Replace the in-memory `verification_sessions` dictionary in `websocket_manager.py` with a Redis-backed implementation.
5.  **Implement Status Endpoint**: Build out the functionality for the `/verification-status/{verification_id}` endpoint.
6.  **Define Data Models**: Create Pydantic models for the data structures passed between agent components (`FactCheckResult`, `VerdictResult`).

**Phase 3: Core Logic Refactoring**

7.  **Decompose `VeritasAgent`**:
    *   Create the new service classes (`VerificationOrchestrator`, `ImageAnalysisService`, etc.).
    *   Incrementally migrate logic from `VeritasAgent` into the new services, using the Pydantic models from the previous step.
    *   Update `VeritasAgent` (or the new `VerificationOrchestrator`) to orchestrate calls to these services.
    *   Update `routers/verification.py` to use the new orchestrator. 
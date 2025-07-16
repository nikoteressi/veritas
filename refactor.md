### Analysis of the Existing Codebase

The current architecture is modular, built around a `VerificationPipeline` that executes a series of steps. This is a solid foundation. The key files for this refactoring are:

* **`agent/pipeline/verification_pipeline.py`**: The central orchestrator. We will redefine the sequence of steps here.
* **`agent/pipeline/pipeline_steps.py`**: Where the logic for each step resides. We will add new steps and heavily modify existing ones.
* **`agent/services/*` and `agent/analyzers/*`**: These contain the core logic for each step. We will refactor these to use different models and inputs.
* **`agent/models/*`**: The Pydantic models that define the data structures passed between steps. New models will be created, and existing ones will be updated.
* **`agent/llm.py`**: The LLM manager. It will be updated to support and manage multiple models.
* **`app/config.py`**: Where we will define the new model configurations.
* **`agent/prompts.yaml`**: This file will be updated with new and modified prompts for the different LLMs.

### The Refactoring Plan

This plan is broken down into three phases to ensure a structured and manageable transition.

---

## Refactoring Plan: Multi-Model Workflow

### Phase 1: Configuration and Data Model Updates

#### 1.1. Configure Multi-Model Support

* **Objective:** Enable the system to use two distinct large language models: a vision model for screenshot parsing and a text/reasoning model for all other tasks.
* **Files to Modify:**
    * `backend/app/config.py`
    * `backend/agent/llm.py`
* **Actions:**
    1.  **Update `Settings` in `app/config.py`:**
        * Rename the existing `ollama_model` setting to `vision_model_name` (e.g., `qwen2.5-vl`).
        * Add a new setting `reasoning_model_name` for the text model.
    2.  **Refactor `OllamaLLMManager` in `agent/llm.py`:**
        * Modify the `__init__` method to initialize two `ChatOllama` instances: `self.vision_llm` and `self.reasoning_llm`, based on the new config settings.
        * Update `invoke_multimodal` to exclusively use `self.vision_llm`.
        * Update `invoke_text_only` to exclusively use `self.reasoning_llm`. This ensures the correct model is used for each task type.

#### 1.2. Define New Pydantic Models

* **Objective:** Create the necessary Pydantic models for the new, structured workflow.
* **Files to Modify:**
    * `backend/agent/models/screenshot_data.py` (This exists and is perfect for the first step).
    * `backend/agent/models/verification_context.py`
* **Actions:**
    1.  **Leverage `ScreenshotData`:** The existing `backend/agent/models/screenshot_data.py` is perfectly suited for the output of the new screenshot parsing step. No changes are needed there.
    2.  **Update `VerificationContext`:** In `verification_context.py`, add new fields to hold the results of the new pipeline steps:
        * `screenshot_data: Optional[ScreenshotData] = None`
        * `fact_hierarchy: Optional[FactHierarchy] = None`
        * `summary: Optional[str] = None`

### Phase 2: Pipeline and Service Refactoring

#### 2.1. Step 1: Create a Dedicated Screenshot Parsing Step

* **Objective:** Implement the first step of the new workflow, which only parses the screenshot into a structured `ScreenshotData` object.
* **Files to Modify:**
    * Create `backend/agent/services/screenshot_parser.py`.
    * Update `backend/agent/pipeline/pipeline_steps.py`.
    * Update `backend/agent/prompts.yaml`.
* **Actions:**
    1.  **Create `ScreenshotParserService`:** This new service will have one method, `parse`, which takes the image bytes, invokes the **vision model** (`qwen2.5-vl`) with a new, specialized prompt, and parses the output into the `ScreenshotData` model.
    2.  **Create `ScreenshotParsingStep`:** In `pipeline_steps.py`, create a new step class that uses the `ScreenshotParserService` and saves the result to `context.screenshot_data`.
    3.  **Create New Prompt:** In `prompts.yaml`, add a `screenshot_parsing` prompt designed to guide the vision model in filling out the fields of the `ScreenshotData` model.

#### 2.2. Step 2: Refactor Temporal Analysis to Use LLM

* **Objective:** Replace the current regex-based temporal analysis with a more robust LLM-based approach.
* **Files to Modify:**
    * `backend/agent/analyzers/temporal_analyzer.py`
    * `backend/agent/prompts.yaml`
* **Actions:**
    1.  **Rewrite `TemporalAnalyzer`:** The `analyze` method will now take the `ScreenshotData` from the context.
    2.  It will format a prompt for the **reasoning model**, providing the extracted text and timestamps.
    3.  The prompt will ask the model to identify the post's creation date, list all other mentioned dates, and describe their relationship.
    4.  Add a `temporal_analysis_llm` prompt to `prompts.yaml`.

#### 2.3. Step 3: Create Post Analysis Step

* **Objective:** Create a new step that analyzes the parsed screenshot data to extract the main thesis and supporting facts.
* **Files to Modify:**
    * Create `backend/agent/services/post_analyzer.py`.
    * Update `backend/agent/pipeline/pipeline_steps.py`.
    * Update `backend/agent/prompts.yaml`.
* **Actions:**
    1.  **Create `PostAnalyzerService`:** This service will take the `ScreenshotData` and `temporal_analysis` results.
    2.  It will use the **reasoning model** to:
        * Generate the `FactHierarchy` (the primary thesis and supporting facts).
        * Determine the post's `primary_topic` (e.g., financial, political).
    3.  **Create `PostAnalysisStep`:** Add a new step class in `pipeline_steps.py` to orchestrate this.
    4.  Add a `post_analysis` prompt to `prompts.yaml`.

#### 2.4. Step 4 & 5: Update Fact-Checking and Create Summarization Step

* **Objective:** Enhance fact-checking with better context and add a dedicated summarization step.
* **Files to Modify:**
    * `backend/agent/services/fact_checking.py`
    * Create `backend/agent/services/summarizer.py`.
    * Update `backend/agent/pipeline/pipeline_steps.py`.
    * Update `backend/agent/prompts.yaml`.
* **Actions:**
    1.  **Update `FactCheckingService`:** Modify the `_generate_contextual_search_queries` method to incorporate the detailed output from the new `TemporalAnalysisStep` to create more precise, time-aware search queries.
    2.  **Create `SummarizerService`:** This service will take the search results and the temporal analysis and use the **reasoning model** to create a concise summary.
    3.  **Create `SummarizationStep`:** Add the corresponding step to the pipeline.
    4.  Add a `summarization` prompt to `prompts.yaml`.

#### 2.5. Step 6 & 7: Refactor Motive Analysis and Final Verdict

* **Objective:** Update the motive analysis and final verdict steps to use the new, richer context.
* **Files to Modify:**
    * `backend/agent/analyzers/motives_analyzer.py`
    * `backend/agent/services/verdict.py`
* **Actions:**
    1.  **Update `MotivesAnalyzer`:** The `analyze` method will now receive the summary from the new `SummarizationStep` in addition to the post data and temporal analysis. The prompt will be updated to reflect this richer input.
    2.  **Update `VerdictService`:** The `generate` method will now also receive the `motives_analysis` result as a key input for generating the final verdict. The corresponding prompt will be updated.

### Phase 3: Finalizing the Pipeline

#### 3.1. Rebuild the Main Pipeline

* **Objective:** Assemble the new, refactored workflow.
* **Files to Modify:**
    * `backend/agent/pipeline/verification_pipeline.py`
    * `backend/app/config.py`
* **Actions:**
    1.  **Update `VerificationSteps` enum in `config.py`** to reflect the new, more granular workflow.
    2.  **Modify the `VerificationPipeline` constructor in `verification_pipeline.py`** to execute the new sequence of steps:
        1.  `ValidationStep`
        2.  `ScreenshotParsingStep` (New)
        3.  `TemporalAnalysisStep` (Refactored)
        4.  `PostAnalysisStep` (New)
        5.  `ReputationRetrievalStep`
        6.  `FactCheckingStep` (Refactored)
        7.  `SummarizationStep` (New)
        8.  `VerdictGenerationStep` (Refactored to include motives)
        9.  `MotivesAnalysisStep` (Refactored)
        10. `ReputationUpdateStep`
        11. `ResultStorageStep`

#### 3.2. Update Final Result Compilation

* **Objective:** Ensure the final API response includes all the new data points.
* **Files to Modify:**
    * `backend/agent/services/result_compiler.py`
    * `backend/app/schemas/api.py`
* **Actions:**
    1.  Update `VerificationResponse` in `api.py` to include new fields like `motives_analysis`, `summary`, etc.
    2.  Update the `compile_result` method in `result_compiler.py` to map all the new data from the `VerificationContext` into the final response object.

This refactoring plan will result in a more robust, accurate, and maintainable system that leverages the strengths of different AI models for their respective tasks.
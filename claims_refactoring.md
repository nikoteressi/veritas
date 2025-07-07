### **Refactoring Plan: "From Monolith to Smart Pipeline"**

**Objective:** To transform the screenshot analysis system from a single, monolithic LLM call into a robust, multi-stage "Locator + Parsers" pipeline. This will increase reliability, simplify debugging, and reduce operational costs.

### **Phase 1: Preparation and Groundwork**

**Goal:** Secure the current functionality and prepare the environment for changes.

1.  **Create a Feature Branch:**
    * `git checkout -b feature/refactor-parsing-pipeline`

2.  **Write an Integration Test:**
    * Create a test that runs the entire current process: it takes a screenshot and verifies that the output is a correct claim hierarchy and a reasonably accurate date.
    * This test is expected to **fail** after the refactoring begins, and our final goal will be to make it pass again.

### **Phase 2: Refactor `temporal_analysis` (Quick Win)**

**Goal:** Replace the brittle date-parsing module with a robust, specialized library.

1.  **Replace the Logic:**
    * In the file `backend/agent/services/temporal_analysis.py`:
        * Delete all existing functions that use regular expressions.
        * Create a single new function, e.g., `parse_timestamp_text(text: str) -> datetime | None`.
        * Inside this function, use the `dateparser` library: `return dateparser.parse(text, settings={'PREFER_DATES_FROM': 'past'})`.

2.  **Handle Errors:**
    * Add basic error handling for cases where `dateparser` returns `None`. Log this event as a `warning`.

### **Phase 3: Evolve `ImageAnalysisService` into a "Locator"**

**Goal:** Modify the core image analysis service so that it **locates** metadata instead of parsing it.

1.  **Update Pydantic Models:**
    * In the file `backend/app/schemas.py`:
        * Create a new model `MetadataSnippets(BaseModel)` with the following fields:
            * `author_text: str | None`
            * `timestamp_text: str | None`
        * Modify the `ImageAnalysisResult` model to include `metadata_snippets: MetadataSnippets`.

2.  **Update the LLM Prompt:**
    * In the file `backend/agent/prompts.py`:
        * Modify the `MULTIMODAL_ANALYSIS_PROMPT`.
        * Instruct the model to return the new JSON structure, including the `metadata_snippets` object.
        * **Crucially:** The prompt must clearly instruct the model to return the *raw text* for the date and author (e.g., "5h ago," not a calculated ISO format).

3.  **Adapt the Service:**
    * In `backend/agent/services/image_analysis.py`:
        * Modify `ImageAnalysisService.analyse` to parse the new JSON response from the LLM and return the updated `ImageAnalysisResult` object.

### **Phase 4: Create and Integrate "Micro-parsers"**

**Goal:** Build small, specialized functions to process the raw metadata snippets provided by the "Locator."

1.  **Create a Parsers Module:**
    * Create a new file: `backend/agent/services/metadata_parsers.py`.
    * Move the updated `parse_timestamp_text` function from `temporal_analysis` into this new file.
    * Create a new function `parse_author_text(text: str) -> str | None` that cleans up the username (e.g., removes the `@` symbol).

2.  **Integrate into the Main Pipeline:**
    * In the agent's main control flow (likely in `backend/agent/agent.py` or a similar file):
        * **Step 1:** Call `ImageAnalysisService.analyse` to get the `ImageAnalysisResult`.
        * **Step 2:** Extract the `metadata_snippets` from the result.
        * **Step 3:** Pass `metadata_snippets.timestamp_text` to `metadata_parsers.parse_timestamp_text`.
        * **Step 4:** Pass `metadata_snippets.author_text` to `metadata_parsers.parse_author_text`.
        * **Step 5:** Assemble the claims and the parsed metadata into a final object for the next stage (e.g., fact-checking).

### **Phase 5: Finalization and Cleanup**

**Goal:** Verify functionality, update documentation, and remove obsolete code.

1.  **Run Tests:**
    * Ensure the integration test created in Phase 1 now passes successfully.
    * Write a few unit tests for the new "micro-parsers."

2.  **Update Documentation:**
    * Edit `README.md` or other project documentation to describe the new parsing architecture.

3.  **Remove Dead Code:**
    * Completely delete the old `temporal_analysis` module and any other unnecessary functions that have been replaced.

4.  **Merge the Branch:**
    * `git merge feature/refactor-parsing-pipeline` into the main development branch.
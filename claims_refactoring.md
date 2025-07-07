# Refactoring Plan: Integrating a Hierarchical Fact Structure

## 1. The Problem

The current implementation extracts facts as a flat list of strings (`claims: List[str]`). This approach fails to capture the logical relationship between facts, leading to loss of context during verification, inefficiency, and unnatural, disjointed final verdicts. Our goal is to fix this without breaking existing functionality that relies on user prompt context and temporal data.

## 2. The Goal

To refactor the system to understand the *story* within the source by evolving from a flat list to a hierarchical structure. This new structure will identify a **primary thesis** and its **supporting facts**. This will be achieved by **integrating** new instructions into our existing prompts, not by replacing them.

## 3. Step-by-Step Refactoring Guide for the AI Agent

### Step 1: Update Pydantic Models (No Changes from Previous Plan)

This step remains the same. The foundation of this refactoring is changing the data structure.

**File to modify:** `backend/app/schemas.py`

**Action:** Replace the `claims: List[str]` field in `ImageAnalysisResult` with a more descriptive `FactHierarchy` model.

**New Code to Implement:**
```python
# in backend/app/schemas.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Fact(BaseModel):
    """Represents a single, atomic, verifiable fact that supports the primary thesis."""
    description: str = Field(description="A clear, concise statement of the fact for verification.")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data extracted for this fact (e.g., amounts, dates, entities) to aid in targeted verification."
    )

class FactHierarchy(BaseModel):
    """Represents the structured, hierarchical understanding of the claims made in the source."""
    primary_thesis: str = Field(description="The single, overarching claim or main point of the source. This summarizes the entire message.")
    supporting_facts: List[Fact] = Field(description="A list of atomic, verifiable facts that support the primary thesis.")

class ImageAnalysisResult(BaseModel):
    """The result of a multimodal analysis of an image."""
    # ... other fields like 'summary', 'status' remain the same

    # REMOVE the old 'claims' field:
    # claims: List[str] = Field(description="A list of verifiable factual claims extracted from the image.")
    
    # ADD the new 'fact_hierarchy' field:
    fact_hierarchy: Optional[FactHierarchy] = Field(None, description="A structured representation of the claims and their relationships.")

````

### Step 2: **Integrate** Hierarchy Extraction into the Existing Multimodal Prompt

**File to modify:** `backend/agent/prompts.py`

**Action:** Modify `MULTIMODAL_ANALYSIS_PROMPT`. Instead of replacing the entire prompt, we will insert the new instructions and the new JSON output format, preserving existing context variables like `user_prompt` and `current_date`.

**Instructions:**

1.  Locate the section in the prompt that describes the analysis task.
2.  Replace the instruction "extract a list of claims" with the new two-step logic: "first, identify the primary thesis; second, extract supporting facts."
3.  Update the section describing the JSON output format to match the new `FactHierarchy` Pydantic model.
4.  Ensure all existing template variables (`{user_prompt}`, `{current_date}`, etc.) remain in their original positions.

**Example of an *Updated* Prompt (Illustrating Integration):**

```python
# in backend/agent/prompts.py

# This is an example of what the MODIFIED prompt should look like.
# DO NOT blindly copy-paste. INTEGRATE these changes into your existing prompt.

MULTIMODAL_ANALYSIS_PROMPT = """
You are an advanced analysis agent. Your task is to analyze the user's request and the provided image based on the current context.

**Current Date for Context:** {current_date}
**Original User Prompt:** {user_prompt}

**Your Analysis Task:**
Analyze the provided image and any accompanying text. Your goal is to deconstruct the information into a hierarchical structure.

1.  **Identify Primary Thesis:** First, determine the single, main point the author is trying to make. This should be a holistic summary of the core message, taking the user's prompt and current date into account for full context.
2.  **Extract Supporting Facts:** Second, extract all the specific, atomic, and verifiable facts that underpin this primary thesis. For each fact, provide a clear `description` and a `context` dictionary with its structured data.

**Output Format:**
You MUST respond ONLY with a JSON object that strictly follows the Pydantic model structure provided below. Do not include any other text, markdown formatting, or explanations.

**JSON OUTPUT STRUCTURE:**
{{
  "fact_hierarchy": {{
    "primary_thesis": "The main point or overarching claim, e.g., 'The post claims BlackRock significantly increased its Bitcoin holdings with a recent purchase.'",
    "supporting_facts": [
      {{
        "description": "BlackRock purchased 4,970 BTC for $530.6M on May 21, 2024.",
        "context": {{
          "entity": "BlackRock",
          "action": "purchased",
          "amount_btc": 4970,
          "value_usd": 530600000,
          "date": "2024-05-21"
        }}
      }},
      {{
        "description": "BlackRock's total holdings are now 643,974 BTC.",
        "context": {{
          "entity": "BlackRock",
          "metric": "total holdings btc",
          "value": 643974
        }}
      }}
    ]
  }},
  "other_existing_fields": "..."
}}
"""
```

### Step 3: Adapt the Fact-Checking Service (No Changes from Previous Plan)

This step is logically sound and does not depend on the prompt's input variables, only on its output structure.

**File to modify:** `backend/agent/services/fact_checking_service.py`

**Action:** Update the logic to use the new `FactHierarchy` structure for creating context-aware search queries.

**Refactoring Logic (pseudocode):**

```python
# in backend/agent/services/fact_checking_service.py

class FactCheckingService:
    def execute(self, analysis_result: ImageAnalysisResult) -> List[CheckedFact]:
        if not analysis_result.fact_hierarchy:
            return []

        fact_hierarchy = analysis_result.fact_hierarchy
        primary_thesis = fact_hierarchy.primary_thesis
        
        checked_facts = []
        for fact in fact_hierarchy.supporting_facts:
            # Generate a context-aware query using both the thesis and the specific fact
            contextual_query = self._generate_contextual_query(primary_thesis, fact)
            search_results = self.search_tool.search(query=contextual_query)
            verification_result = self._verify_fact(fact, search_results)
            checked_facts.append(verification_result)
            
        return checked_facts
```

### Step 4: Adapt the Verdict Synthesis Service (No Changes from Previous Plan)

This step correctly focuses on synthesizing a final answer from the new structured input. It's an enhancement that doesn't conflict with existing context.

**File to modify:** `backend/agent/services/verdict_service.py`

**Action:** Update the verdict generation prompt to synthesize a narrative conclusion from the hierarchy of verified facts.

**New Verdict Prompt Logic:**

```python
# in backend/agent/services/verdict_service.py

def generate_verdict_prompt(primary_thesis: str, checked_facts: List[CheckedFact], original_user_prompt: str) -> str:
    prompt = f"""
    You are a professional fact-checking analyst. Your task is to write a final, synthesized verdict that directly answers the user's original question.

    **User's Original Question:** "{original_user_prompt}"
    **The Primary Thesis of the Source Material:** "{primary_thesis}"

    **Evidence Report:**
    We have verified the facts supporting the thesis. Here is the summary:
    """
    for fact in checked_facts:
        prompt += f"- Fact: '{fact.description}' | Verification Result: {fact.is_true} | Evidence: {fact.summary}\n"

    prompt += """
    \n**Final Verdict Instructions:**
    Based on all the information above, write a clear and definitive final verdict.
    1.  Start with a conclusive rating: **True**, **False**, **Partially True**, or **Misleading**.
    2.  Write a concise, easy-to-read paragraph that directly addresses the user's original question.
    3.  Synthesize the evidence into a coherent narrative. Explain *why* the source's main thesis is correct or incorrect based on the verified facts. Do not just list the facts again.
    """
    return prompt
# Refactoring Plan v2: Robust Prompt Management

This plan refactors the application to manage prompts as validated, structured assets, completely decoupled from application logic and configuration.

### Step 1: Define Pydantic Models for Prompt Structures

**Action:** CREATE
**File Path:** `backend/agent/models.py`

**Content:**
```python
from pydantic import BaseModel, RootModel
from typing import List, Literal
from langchain_core.prompts import ChatPromptTemplate

# Defines the allowed roles to prevent typos
Role = Literal["system", "human", "ai", "tool"]

class MessageTemplate(BaseModel):
    role: Role
    template: str

class PromptStructure(RootModel[List[MessageTemplate]]):
    """
    A Pydantic RootModel to validate a list of MessageTemplate objects,
    representing a full chat prompt structure.
    """
    def to_chat_prompt_template(self) -> ChatPromptTemplate:
        """Helper method to convert the validated structure to a LangChain ChatPromptTemplate."""
        messages = [(item.role, item.template) for item in self.root]
        return ChatPromptTemplate.from_messages(messages)
````

### Step 2: Create a Structured `prompts.yaml` File

**Action:** CREATE
**File Path:** `backend/agent/prompts.yaml`

**Content:**

````yaml
multimodal_analysis:
  - role: system
    template: |
      You are an expert AI analyst specializing in social media content verification. Your task is to analyze content (images and text) for authenticity and potential manipulation.
      **Current Date for Context:** {current_date}
      **Output Format:** You must respond with a JSON object that strictly follows this structure: {format_instructions}
  - role: human
    template: |
      Analyze the following content.
      **User Query:** "{user_query}"
      **Image Path:** "{image_path}"

verdict_generation:
  - role: system
    template: |
      You are a professional fact-checking analyst. Your role is to synthesize the results from various analysis steps and provide a final, well-reasoned verdict.
      **Output Format:** You must respond with a JSON object that strictly follows this structure: {format_instructions}
  - role: human
    template: |
      Review the collected evidence and provide your final verdict and justification.
      **Analysis State:**
      ```json
      {state}
      ```
````

### Step 3: Add `PyYAML` to Dependencies

**Action:** MODIFY
**File Path:** `backend/requirements.txt`

**Change:**
Add `PyYAML` to the list of requirements if it's not already present.

### Step 4: Create a Dedicated `PromptManager`

**Action:** CREATE
**File Path:** `backend/agent/prompt_manager.py`

**Content:**

```python
import yaml
from pathlib import Path
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from .models import PromptStructure

class PromptManager:
    """
    Loads, validates, and serves chat prompt templates from a YAML file.
    """
    def __init__(self, file_path: str = "agent/prompts.yaml"):
        root_dir = Path(__file__).parent.parent
        self.prompts_path = root_dir / file_path
        self._raw_prompts = self._load_raw_prompts()
        self._validated_prompts = self._validate_prompts()

    def _load_raw_prompts(self) -> Dict:
        """Loads the raw YAML file."""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Prompts file not found at: {self.prompts_path}")

    def _validate_prompts(self) -> Dict[str, PromptStructure]:
        """Validates the raw prompt data against Pydantic models."""
        validated = {}
        for name, data in self._raw_prompts.items():
            try:
                validated[name] = PromptStructure.model_validate(data)
            except Exception as e:
                raise ValueError(f"Invalid prompt structure for '{name}': {e}")
        return validated

    def get_prompt_template(self, name: str) -> ChatPromptTemplate:
        """Gets a validated prompt and returns it as a ChatPromptTemplate instance."""
        prompt_structure = self._validated_prompts.get(name)
        if not prompt_structure:
            raise ValueError(f"Prompt template '{name}' not found or is invalid.")
        return prompt_structure.to_chat_prompt_template()

# Singleton instance to ensure prompts are loaded and validated only once.
prompt_manager = PromptManager()
```

### Step 5: Refactor Services to Use the `PromptManager`

**Action:** MODIFY
**Files:** `backend/agent/services/image_analysis.py`, `backend/agent/services/fact_checking.py`

**Change:**
Update the services to get pre-built `ChatPromptTemplate` objects from the `prompt_manager`.

**Example for `image_analysis.py`:**

```python
# Other imports...
from agent.prompt_manager import prompt_manager
from datetime import datetime

class ImageAnalysisService:
    def __init__(self, llm, parser):
        self.prompt_template = prompt_manager.get_prompt_template("multimodal_analysis")
        self.chain = self.prompt_template | llm | parser
        self.parser = parser

    def analyze_image(self, image_path: str, user_query: str):
        current_date = datetime.now().strftime("%Y-%m-%d")
        return self.chain.invoke({
            "current_date": current_date,
            "user_query": user_query,
            "image_path": image_path,
            "format_instructions": self.parser.get_format_instructions(),
        })
```

*(Apply a similar refactoring to `FactCheckingService` using the `"verdict_generation"` template).*

### Step 6: Delete Obsolete File

**Action:** DELETE
**File Path:** `backend/agent/prompts.py`

**Change:**
Remove the old, hardcoded prompts file as it is now redundant.

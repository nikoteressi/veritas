import yaml
from pathlib import Path
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from .models.prompt_structures import PromptStructure

class PromptManager:
    """
    Loads, validates, and serves chat prompt templates and other prompt-related
    configurations from a YAML file.
    """
    def __init__(self, file_path: str = "prompts.yaml"):
        self.prompts_path = Path(__file__).parent / file_path
        self._raw_data = self._load_raw_data()
        self.domain_specific_descriptions = self._raw_data.get("domain_specific_descriptions", {})
        self._validated_prompts = self._validate_prompts()

    def _load_raw_data(self) -> Dict:
        """Loads the raw YAML file."""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Prompts file not found at: {self.prompts_path}")

    def _validate_prompts(self) -> Dict[str, PromptStructure]:
        """Validates the raw prompt data against Pydantic models."""
        validated = {}
        # Iterate over items that are not the domain descriptions
        for name, data in self._raw_data.items():
            if name == "domain_specific_descriptions":
                continue
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
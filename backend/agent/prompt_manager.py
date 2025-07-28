from pathlib import Path

import yaml
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
        self.domain_specific_descriptions = self._raw_data.get(
            "domain_specific_descriptions", {}
        )
        self._validated_prompts = self._validate_prompts()

    def _load_raw_data(self) -> dict:
        """Loads the raw YAML file."""
        try:
            with open(self.prompts_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Prompts file not found at: {self.prompts_path}") from e

    def _validate_prompts(self) -> dict[str, PromptStructure]:
        """Validates the raw prompt data against Pydantic models."""
        validated = {}
        # Iterate over items that are not the domain descriptions
        for name, data in self._raw_data.items():
            if name == "domain_specific_descriptions":
                continue
            try:
                validated[name] = PromptStructure.model_validate(data)
            except Exception as e:
                raise ValueError(
                    f"Invalid prompt structure for '{name}': {e}") from e
        return validated

    def get_prompt_template(self, name: str) -> ChatPromptTemplate:
        """Gets a validated prompt and returns it as a ChatPromptTemplate instance."""
        prompt_structure = self._validated_prompts.get(name)
        if not prompt_structure:
            raise ValueError(
                f"Prompt template '{name}' not found or is invalid.")
        return prompt_structure.to_chat_prompt_template()

    def get_prompt(self, name: str, **kwargs) -> str:
        """Gets a prompt template and formats it with the provided parameters."""
        prompt_template = self.get_prompt_template(name)
        try:
            # Format the prompt with the provided parameters
            formatted_messages = prompt_template.format_messages(**kwargs)
            # Combine all messages into a single string
            return "\n\n".join([msg.content for msg in formatted_messages])
        except KeyError as e:
            raise ValueError(
                f"Missing required parameter for prompt '{name}': {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to format prompt '{name}': {e}") from e

    def get_domain_role_description(self, domain: str) -> str:
        """
        Get domain-specific role description for fact-checking.

        Args:
            domain: The domain name (e.g., 'financial', 'medical', 'political', 'scientific')

        Returns:
            Domain-specific role description or general description if domain not found
        """
        # Normalize domain name to match keys in domain_specific_descriptions
        normalized_domain = domain.lower()

        # Map common domain variations to standard keys
        domain_mapping = {
            "finance": "financial",
            "bitcoin": "financial",
            "cryptocurrency": "financial",
            "crypto": "financial",
            "investment": "financial",
            "market": "financial",
            "trading": "financial",
            "health": "medical",
            "medicine": "medical",
            "politics": "political",
            "government": "political",
            "science": "scientific",
            "research": "scientific",
        }

        # Get the standard domain key
        standard_domain = domain_mapping.get(
            normalized_domain, normalized_domain)

        # Get domain-specific description or fallback to general
        role_description = self.domain_specific_descriptions.get(
            standard_domain,
            "You are a versatile fact-checker that verifies claims by cross-referencing "
            "reputable sources and established fact-checking organizations.",
        )

        return role_description


# Singleton instance to ensure prompts are loaded and validated only once.
prompt_manager = PromptManager()

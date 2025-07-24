from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, RootModel

# Defines the allowed roles to prevent typos
Role = Literal["system", "human", "ai", "tool"]


class MessageTemplate(BaseModel):
    role: Role
    template: str


class PromptStructure(RootModel[list[MessageTemplate]]):
    """
    A Pydantic RootModel to validate a list of MessageTemplate objects,
    representing a full chat prompt structure.
    """

    def to_chat_prompt_template(self) -> ChatPromptTemplate:
        """Helper method to convert the validated structure to a LangChain ChatPromptTemplate."""
        messages = [(item.role, item.template) for item in self.root]
        return ChatPromptTemplate.from_messages(messages)

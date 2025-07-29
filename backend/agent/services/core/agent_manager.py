"""
Service for managing LLM agents and their configuration.
"""

from __future__ import annotations

import logging

from app.exceptions import AgentError
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from ...llm import llm_manager
from ...tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages LLM agent initialization and configuration."""

    def __init__(self):
        """Initialize AgentManager without performing I/O operations."""
        self.llm = None
        self.tools = AVAILABLE_TOOLS
        self.agent_executor: AgentExecutor | None = None
        self._initialized = False

    async def _initialize_agent(self):
        """Initialize the LangChain agent executor asynchronously."""
        try:
            # Initialize LLM if not already done
            if self.llm is None:
                self.llm = llm_manager.reasoning_llm

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful AI assistant with access to tools for fact-checking.",
                    ),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate",
            )

            self._initialized = True
            logger.info("Agent executor initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize agent: %s", e)
            raise AgentError(f"Failed to initialize agent: {e}") from e

    def get_agent_executor(self) -> AgentExecutor:
        """Get the initialized agent executor."""
        if not self._initialized or self.agent_executor is None:
            raise RuntimeError("AgentManager not initialized. Call create_agent_manager() first.")
        return self.agent_executor

    def is_agent_ready(self) -> bool:
        """Check if the agent is ready for use."""
        return self._initialized and self.agent_executor is not None


async def create_agent_manager() -> AgentManager:
    """
    Async factory function to create and initialize an AgentManager.

    Returns:
        Initialized AgentManager instance
    """
    logger.info("Creating and initializing AgentManager...")

    try:
        manager = AgentManager()
        await manager._initialize_agent()
        logger.info("AgentManager created and initialized successfully")
        return manager
    except Exception as e:
        logger.error("Failed to create AgentManager: %s", e)
        raise AgentError(f"Failed to create AgentManager: {e}") from e


def get_agent_manager_from_app(app) -> AgentManager:
    """
    Get the AgentManager instance from FastAPI app state.

    Args:
        app: FastAPI application instance

    Returns:
        AgentManager instance

    Raises:
        RuntimeError: If AgentManager is not initialized in app state
    """
    if not hasattr(app.state, "agent_manager"):
        raise RuntimeError("AgentManager not found in app state. Ensure the app was started properly.")

    agent_manager = app.state.agent_manager
    if not agent_manager.is_agent_ready():
        raise RuntimeError("AgentManager is not ready. Initialization may have failed.")

    return agent_manager

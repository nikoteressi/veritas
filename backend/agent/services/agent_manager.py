"""
Service for managing LLM agents and their configuration.
"""
import logging
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from agent.llm import llm_manager
from agent.tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages LLM agent initialization and configuration."""
    
    def __init__(self):
        self.llm = llm_manager.llm
        self.tools = AVAILABLE_TOOLS
        self.agent_executor: Optional[AgentExecutor] = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent executor."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant with access to tools for fact-checking."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            logger.info("Agent executor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def get_agent_executor(self) -> AgentExecutor:
        """Get the initialized agent executor."""
        if self.agent_executor is None:
            self._initialize_agent()
        return self.agent_executor
    
    def is_agent_ready(self) -> bool:
        """Check if the agent is ready for use."""
        return self.agent_executor is not None


# Singleton instance
agent_manager = AgentManager() 
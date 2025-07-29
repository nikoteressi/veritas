"""Tool registry for managing available agent tools."""

from typing import Dict, List
from langchain.tools import BaseTool

from .search import searxng_tool


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        self.register_tool(searxng_tool)

    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())


# Global registry instance
tool_registry = ToolRegistry()

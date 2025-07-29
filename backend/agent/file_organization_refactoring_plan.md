# File Organization Refactoring Plan

**Estimated Time**: 1 day  
**Priority**: High  
**Risk Level**: Medium (requires careful import management)

## Executive Summary

This refactoring plan reorganizes the `backend/agent` directory to improve code maintainability, logical grouping, and developer experience. The current structure has files scattered at the root level that should be organized into logical subdirectories based on their functionality.

## Current State Analysis

### Files Requiring Reorganization

The following files are currently in `backend/agent/` root and need to be moved:

1. **Storage/Database Layer**:
   - `chroma_client.py` - ChromaDB client implementation
   - `vector_store.py` - Vector store operations

2. **LLM/AI Layer**:
   - `llm.py` - Ollama LLM manager with global instance `llm_manager`
   - `ollama_embeddings.py` - Ollama embeddings implementation

3. **Configuration/Prompts**:
   - `prompt_manager.py` - Prompt management system
   - `prompts.yaml` - Prompt templates

4. **Tools**:
   - `tools.py` - SearxNG search tool with global instances `searxng_tool` and `AVAILABLE_TOOLS`

5. **Orchestration**:
   - `workflow_coordinator.py` - Workflow coordination with global instance `workflow_coordinator`

### Error Handling Classes

**Primary Error Handling Location**: `backend/app/error_handlers.py`
- Contains async functions for handling all exception types
- Includes `setup_error_handlers()` function for FastAPI integration
- Maps exceptions to handlers via `EXCEPTION_HANDLERS` dictionary

**Custom Exceptions**: `backend/app/exceptions.py`
- Base class: `VeritasException`
- Specialized exceptions: `AgentError`, `ToolError`, `LLMError`, etc.

### Current Dependencies Analysis

Based on import analysis, the following files have dependencies on the files to be moved:

#### Dependencies on `llm.py`:
- `backend/agent/services/output/verdict.py`
- `backend/agent/services/graph/graph_builder.py`
- `backend/app/services/verification_service.py`
- `backend/agent/analyzers/temporal_analyzer.py`
- `backend/agent/pipeline/graph_fact_checking_step.py`

#### Dependencies on `tools.py`:
- `backend/agent/services/output/verdict.py`
- `backend/agent/services/graph/graph_builder.py`
- `backend/app/services/verification_service.py`

#### Dependencies on `workflow_coordinator.py`:
- `backend/app/services/verification_service.py`

#### Dependencies on other files:
- Multiple files import from `chroma_client`, `vector_store`, `prompt_manager`, `ollama_embeddings`

## Proposed New Directory Structure

```
backend/agent/
├── clients/                    # External service clients
│   ├── __init__.py
│   ├── chroma_client.py       # ChromaDB client
│   └── vector_store.py        # Vector store operations
├── llm/                       # LLM and AI-related components
│   ├── __init__.py
│   ├── manager.py             # Renamed from llm.py
│   └── embeddings.py          # Renamed from ollama_embeddings.py
├── prompts/                   # Prompt management
│   ├── __init__.py
│   ├── manager.py             # Renamed from prompt_manager.py
│   └── templates.yaml         # Renamed from prompts.yaml
├── tools/                     # Agent tools
│   ├── __init__.py
│   ├── search.py              # Renamed from tools.py
│   └── registry.py            # Tool registry (new)
├── orchestration/             # Workflow and coordination
│   ├── __init__.py
│   └── coordinator.py         # Renamed from workflow_coordinator.py
├── analyzers/                 # Existing - no changes
├── models/                    # Existing - no changes
├── pipeline/                  # Existing - no changes
└── services/                  # Existing - no changes
```

## Detailed Implementation Plan

### Phase 1: Create New Directory Structure (30 minutes)

1. **Create new directories**:
   ```bash
   mkdir -p backend/agent/clients
   mkdir -p backend/agent/llm
   mkdir -p backend/agent/prompts
   mkdir -p backend/agent/tools
   mkdir -p backend/agent/orchestration
   ```

2. **Create `__init__.py` files** for each new directory with appropriate exports

### Phase 2: Move and Rename Files (1 hour)

#### 2.1 Clients Directory
- Move `chroma_client.py` → `clients/chroma_client.py`
- Move `vector_store.py` → `clients/vector_store.py`
- Create `clients/__init__.py`:
  ```python
  """Client modules for external services."""
  
  from .chroma_client import ChromaClient, chroma_client
  from .vector_store import VectorStore, vector_store
  
  __all__ = [
      "ChromaClient",
      "chroma_client", 
      "VectorStore",
      "vector_store"
  ]
  ```

#### 2.2 LLM Directory
- Move `llm.py` → `llm/manager.py`
- Move `ollama_embeddings.py` → `llm/embeddings.py`
- Create `llm/__init__.py`:
  ```python
  """LLM and AI-related components."""
  
  from .manager import OllamaLLMManager, llm_manager
  from .embeddings import OllamaEmbeddings, ollama_embeddings
  
  __all__ = [
      "OllamaLLMManager",
      "llm_manager",
      "OllamaEmbeddings", 
      "ollama_embeddings"
  ]
  ```

#### 2.3 Prompts Directory
- Move `prompt_manager.py` → `prompts/manager.py`
- Move `prompts.yaml` → `prompts/templates.yaml`
- Create `prompts/__init__.py`:
  ```python
  """Prompt management system."""
  
  from .manager import PromptManager, prompt_manager
  
  __all__ = [
      "PromptManager",
      "prompt_manager"
  ]
  ```

#### 2.4 Tools Directory
- Move `tools.py` → `tools/search.py`
- Create `tools/registry.py` for tool management
- Create `tools/__init__.py`:
  ```python
  """Agent tools and utilities."""
  
  from .search import SearxNGSearchTool, searxng_tool, AVAILABLE_TOOLS
  from .registry import ToolRegistry
  
  __all__ = [
      "SearxNGSearchTool",
      "searxng_tool",
      "AVAILABLE_TOOLS",
      "ToolRegistry"
  ]
  ```

#### 2.5 Orchestration Directory
- Move `workflow_coordinator.py` → `orchestration/coordinator.py`
- Create `orchestration/__init__.py`:
  ```python
  """Workflow orchestration components."""
  
  from .coordinator import WorkflowCoordinator, workflow_coordinator
  
  __all__ = [
      "WorkflowCoordinator",
      "workflow_coordinator"
  ]
  ```

### Phase 3: Update Import Statements (3-4 hours)

This is the most critical phase requiring systematic updates across all dependent files.

#### 3.1 Update imports in `backend/agent/services/output/verdict.py`
**Current imports to update**:
```python
# OLD
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool

# NEW
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool
```

#### 3.2 Update imports in `backend/agent/services/graph/graph_builder.py`
**Current imports to update**:
```python
# OLD
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool

# NEW  
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool
```

#### 3.3 Update imports in `backend/app/services/verification_service.py`
**Current imports to update**:
```python
# OLD
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool
from backend.agent.workflow_coordinator import workflow_coordinator

# NEW
from backend.agent.llm import llm_manager
from backend.agent.tools import searxng_tool
from backend.agent.orchestration import workflow_coordinator
```

#### 3.4 Update imports in `backend/agent/analyzers/temporal_analyzer.py`
**Current imports to update**:
```python
# OLD
from backend.agent.llm import llm_manager

# NEW
from backend.agent.llm import llm_manager
```

#### 3.5 Update imports in `backend/agent/pipeline/graph_fact_checking_step.py`
**Current imports to update**:
```python
# OLD
from backend.agent.llm import llm_manager

# NEW
from backend.agent.llm import llm_manager
```

#### 3.6 Update all other files with dependencies
Search and replace patterns for remaining files:
- `from backend.agent.chroma_client` → `from backend.agent.clients.chroma_client`
- `from backend.agent.vector_store` → `from backend.agent.clients.vector_store`
- `from backend.agent.prompt_manager` → `from backend.agent.prompts.manager`
- `from backend.agent.ollama_embeddings` → `from backend.agent.llm.embeddings`

### Phase 4: Update Configuration Files (30 minutes)

#### 4.1 Update `prompts/manager.py`
Update the YAML file path reference:
```python
# OLD
PROMPTS_FILE = "prompts.yaml"

# NEW  
PROMPTS_FILE = "prompts/templates.yaml"
```

#### 4.2 Update any configuration that references file paths

### Phase 5: Create Tool Registry (1 hour)

Create `tools/registry.py` for better tool management:
```python
"""Tool registry for managing available agent tools."""

from typing import Dict, List, Type
from langchain.tools import BaseTool

from .search import SearxNGSearchTool, searxng_tool


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
```

## Testing Strategy

### Phase 6: Validation and Testing (2 hours)

#### 6.1 Import Validation
1. **Static Analysis**: Run import checks on all Python files
   ```bash
   python -m py_compile backend/agent/**/*.py
   ```

2. **Import Testing**: Create test script to verify all imports work:
   ```python
   # test_imports.py
   try:
       from backend.agent.llm import llm_manager
       from backend.agent.tools import searxng_tool, AVAILABLE_TOOLS
       from backend.agent.orchestration import workflow_coordinator
       from backend.agent.clients import chroma_client, vector_store
       from backend.agent.prompts import prompt_manager
       print("✅ All imports successful")
   except ImportError as e:
       print(f"❌ Import error: {e}")
   ```

#### 6.2 Functionality Testing
1. **Unit Tests**: Run existing unit tests to ensure no functionality is broken
2. **Integration Tests**: Test key workflows that use the moved components
3. **End-to-End Tests**: Run full verification pipeline to ensure everything works

#### 6.3 Error Handling Validation
1. Verify error handling still works correctly with new import paths
2. Test that all custom exceptions are properly caught and handled
3. Ensure error handlers in `backend/app/error_handlers.py` still function correctly

## Risk Mitigation

### Backup Strategy
1. **Git Branch**: Create feature branch before starting refactoring
2. **Incremental Commits**: Commit after each phase for easy rollback
3. **Testing Checkpoints**: Validate functionality after each major change

### Rollback Plan
If issues arise:
1. **Phase-by-phase rollback**: Revert commits in reverse order
2. **Import mapping**: Keep detailed log of all import changes for quick reversal
3. **Functionality verification**: Test core features after any rollback

## Success Criteria

### Completion Checklist
- [ ] All files moved to appropriate directories
- [ ] All import statements updated correctly
- [ ] No broken imports or missing modules
- [ ] All existing functionality preserved
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Error handling works correctly
- [ ] Global instances (`llm_manager`, `searxng_tool`, `workflow_coordinator`) accessible
- [ ] Configuration files updated
- [ ] Documentation updated

### Quality Metrics
- **Zero broken imports**: All Python files compile successfully
- **100% test coverage maintained**: No reduction in test coverage
- **Performance maintained**: No performance degradation
- **Error handling preserved**: All error scenarios still handled correctly

## Post-Refactoring Benefits

1. **Improved Organization**: Logical grouping of related functionality
2. **Better Maintainability**: Easier to locate and modify specific components
3. **Enhanced Developer Experience**: Clearer code structure and navigation
4. **Scalability**: Better foundation for future feature additions
5. **Reduced Coupling**: Cleaner separation of concerns

## Implementation Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| 1 | 30 min | Create directory structure |
| 2 | 1 hour | Move and rename files |
| 3 | 3-4 hours | Update all import statements |
| 4 | 30 min | Update configuration files |
| 5 | 1 hour | Create tool registry |
| 6 | 2 hours | Testing and validation |
| **Total** | **8 hours** | **Complete refactoring** |

## Notes

- This refactoring maintains all existing functionality
- No fallback implementations are created - only new, properly organized structure
- All dependencies and imports are correctly updated
- Error handling classes remain in their current location (`backend/app/`) as they serve the entire application
- Global instances are preserved and accessible through the new import paths
- The refactoring improves code organization without changing any business logic

---

**Author**: AI Assistant  
**Date**: Current  
**Version**: 1.0  
**Status**: Ready for Implementation
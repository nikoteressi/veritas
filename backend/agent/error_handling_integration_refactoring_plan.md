# Error Handling Integration Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to improve error handling integration across the `backend/agent` directory. The goal is to replace generic `except Exception as e:` patterns with specific exception types that leverage the existing centralized error handling system in `backend/app/error_handlers.py`.

**Estimated Effort:** 1-2 days  
**Priority:** Critical (Priority 1)  
**Risk Level:** Low (incremental changes with thorough testing)

## Current State Analysis

### ✅ Existing Infrastructure (Already Complete)
- **Centralized Error Handling**: Comprehensive system in `backend/app/error_handlers.py`
- **Exception Types**: Well-defined custom exceptions in `backend/app/exceptions.py`
- **Logging Configuration**: Standardized logging via `backend/logging.conf`
- **FastAPI Integration**: Error handlers properly registered in the application

### ⚠️ Issues Identified
- **40+ files** in `backend/agent/` directory use generic `except Exception as e:` patterns
- **25+ files** identified in code analysis report with inconsistent error handling
- **Missing integration** between agent services and centralized error handling
- **Inconsistent error reporting** across different service layers

### ✅ Files Already Compliant
These files already use appropriate specific exceptions and serve as good examples:
- `llm.py` - correctly uses `LLMError`
- `tools.py` - correctly uses `ToolError`  
- `vector_store.py` - correctly uses `VectorStoreError`

## Detailed Implementation Plan

### Phase 1: Extend Exception Types and Handlers (Day 1 Morning - 2 hours)

#### 1.1 Add Missing Exception Types
**File:** `backend/app/exceptions.py`

Add the following new exception types:

```python
class CacheError(VeritasException):
    """Exception raised for cache operations."""
    pass

class EmbeddingError(VeritasException):
    """Exception raised for embedding operations."""
    pass

class GraphError(VeritasException):
    """Exception raised for graph operations."""
    pass

class RelevanceError(VeritasException):
    """Exception raised for relevance scoring operations."""
    pass

class PipelineError(VeritasException):
    """Exception raised for pipeline execution errors."""
    pass

class AnalysisError(VeritasException):
    """Exception raised for general analysis operations."""
    pass
```

#### 1.2 Add Error Handlers
**File:** `backend/app/error_handlers.py`

Add handlers for new exception types:

```python
async def cache_error_handler(request: Request, exc: CacheError) -> JSONResponse:
    """Handle cache errors."""
    logger.error(f"Cache error: {exc.message}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Cache service error",
            "message": "The caching service encountered an issue. Some operations may be slower.",
            "error_code": exc.error_code or "CACHE_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    """Handle embedding errors."""
    logger.error(f"Embedding error: {exc.message}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Embedding service error",
            "message": "The embedding service is temporarily unavailable.",
            "error_code": exc.error_code or "EMBEDDING_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

async def graph_error_handler(request: Request, exc: GraphError) -> JSONResponse:
    """Handle graph errors."""
    logger.error(f"Graph error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Graph operation error",
            "message": "Graph verification encountered an error.",
            "error_code": exc.error_code or "GRAPH_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

async def relevance_error_handler(request: Request, exc: RelevanceError) -> JSONResponse:
    """Handle relevance errors."""
    logger.error(f"Relevance error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Relevance scoring error",
            "message": "Relevance analysis encountered an error.",
            "error_code": exc.error_code or "RELEVANCE_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
    """Handle pipeline errors."""
    logger.error(f"Pipeline error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Pipeline execution error",
            "message": "The verification pipeline encountered an error.",
            "error_code": exc.error_code or "PIPELINE_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    """Handle analysis errors."""
    logger.error(f"Analysis error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Analysis error",
            "message": "Content analysis encountered an error.",
            "error_code": exc.error_code or "ANALYSIS_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
```

#### 1.3 Update Exception Handler Mapping
**File:** `backend/app/error_handlers.py`

Update the `EXCEPTION_HANDLERS` dictionary:

```python
EXCEPTION_HANDLERS = {
    VeritasException: veritas_exception_handler,
    ImageProcessingError: image_processing_error_handler,
    LLMError: llm_error_handler,
    DatabaseError: database_error_handler,
    WebSocketError: websocket_error_handler,
    ValidationError: validation_error_handler,
    ServiceUnavailableError: service_unavailable_error_handler,
    AgentError: agent_error_handler,
    ToolError: tool_error_handler,
    CacheError: cache_error_handler,
    EmbeddingError: embedding_error_handler,
    GraphError: graph_error_handler,
    RelevanceError: relevance_error_handler,
    PipelineError: pipeline_error_handler,
    AnalysisError: analysis_error_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: http_exception_handler,
    Exception: general_exception_handler,
}
```

### Phase 2: Update Core Infrastructure Files (Day 1 Afternoon - 3 hours)

#### 2.1 Files to Update

| File | Current Exceptions | Target Exception | Priority |
|------|-------------------|------------------|----------|
| `chroma_client.py` | 1 generic | `VectorStoreError` | High |
| `ollama_embeddings.py` | 3 generic | `LLMError` | High |
| `prompt_manager.py` | 1 generic | `ValidationError` | Medium |
| `workflow_coordinator.py` | 1 generic | `AgentError` | High |

#### 2.2 Implementation Pattern

**Before:**
```python
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return None
```

**After:**
```python
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise SpecificError(f"Operation failed: {e}") from e
```

#### 2.3 Special Considerations

**Graceful Degradation Pattern** (preserve where appropriate):
```python
except Exception as e:
    logger.warning(f"Optional operation failed, continuing: {e}")
    return default_value  # Keep this pattern for non-critical operations
```

### Phase 3: Update Service Layer Files (Day 2 Morning - 4 hours)

#### 3.1 Service Categories and Target Exceptions

| Service Category | Files | Target Exception | Count |
|------------------|-------|------------------|-------|
| **Relevance Services** | `services/relevance/*.py` | `RelevanceError` | 8 files |
| **Core Services** | `services/core/*.py` | `AgentError` | 6 files |
| **Analysis Services** | `services/analysis/*.py` | `AnalysisError` | 4 files |
| **Graph Services** | `services/graph/*.py` | `GraphError` | 10+ files |
| **Cache Services** | `services/cache/*.py` | `CacheError` | 5 files |
| **Infrastructure Services** | `services/infrastructure/*.py` | `ServiceUnavailableError` | 4 files |

#### 3.2 High-Priority Files (Update First)

1. **`services/relevance/relevance_orchestrator.py`** - 8 generic exceptions
2. **`services/core/system_health_monitor.py`** - 9 generic exceptions  
3. **`services/graph/graph_fact_checking.py`** - 20+ generic exceptions
4. **`services/cache/intelligent_cache.py`** - 13 generic exceptions
5. **`services/analysis/relationship_analysis.py`** - 5 generic exceptions

#### 3.3 Import Updates Required

All service files will need imports added:

```python
from app.exceptions import (
    AgentError,
    CacheError,
    EmbeddingError,
    GraphError,
    RelevanceError,
    AnalysisError,
    ServiceUnavailableError,
)
```

### Phase 4: Update Pipeline and Analyzer Files (Day 2 Afternoon - 3 hours)

#### 4.1 Pipeline Files

| File | Current Exceptions | Target Exception |
|------|-------------------|------------------|
| `pipeline/verification_pipeline.py` | 7 generic | `PipelineError` |
| `pipeline/base_step.py` | 1 generic | `PipelineError` |
| `pipeline/graph_fact_checking_step.py` | 1 generic | `PipelineError` |

#### 4.2 Analyzer Files

| File | Current Exceptions | Target Exception |
|------|-------------------|------------------|
| `analyzers/motives_analyzer.py` | 1 generic | `MotivesAnalysisError` |
| `analyzers/temporal_analyzer.py` | 1 generic | `TemporalAnalysisError` |

#### 4.3 Special Pipeline Considerations

Pipeline files may need to catch and re-raise exceptions from other services:

```python
try:
    result = await some_service.process()
except (GraphError, RelevanceError, AnalysisError) as e:
    logger.error(f"Pipeline step failed: {e}")
    raise PipelineError(f"Pipeline step failed: {e}") from e
except Exception as e:
    logger.error(f"Unexpected pipeline error: {e}")
    raise PipelineError(f"Unexpected pipeline error: {e}") from e
```

## File-by-File Implementation Guide

### Critical Files (Day 1)

#### `chroma_client.py`
- **Current:** 1 generic exception
- **Target:** `VectorStoreError`
- **Import:** `from app.exceptions import VectorStoreError`

#### `ollama_embeddings.py`  
- **Current:** 3 generic exceptions
- **Target:** `LLMError`
- **Import:** `from app.exceptions import LLMError`

#### `workflow_coordinator.py`
- **Current:** 1 generic exception  
- **Target:** `AgentError`
- **Import:** `from app.exceptions import AgentError`

### High-Impact Service Files (Day 2 Morning)

#### `services/relevance/relevance_orchestrator.py`
- **Current:** 8 generic exceptions
- **Target:** `RelevanceError`
- **Special:** May need to catch `EmbeddingError` and re-raise as `RelevanceError`

#### `services/graph/graph_fact_checking.py`
- **Current:** 20+ generic exceptions
- **Target:** `GraphError`
- **Special:** Large file, update incrementally by method

#### `services/core/system_health_monitor.py`
- **Current:** 9 generic exceptions
- **Target:** `AgentError`
- **Special:** Health checks may use graceful degradation

### Pipeline Files (Day 2 Afternoon)

#### `pipeline/verification_pipeline.py`
- **Current:** 7 generic exceptions
- **Target:** `PipelineError`
- **Special:** May catch multiple exception types and re-raise as `PipelineError`

## Testing and Validation Strategy

### 1. Automated Testing
- Run existing test suite after each phase
- Ensure all tests continue to pass
- Add new tests for exception handling paths

### 2. Integration Testing
- Test error responses through FastAPI endpoints
- Verify proper error codes and messages
- Ensure exception chaining preserves context

### 3. Manual Testing
- Test error scenarios in development environment
- Verify graceful degradation still works
- Check error logging and monitoring

### 4. Rollback Plan
- Keep detailed change log for each file
- Test rollback procedures
- Maintain backup of original files

## Risk Mitigation

### Low Risk Factors
- ✅ Existing centralized error handling infrastructure
- ✅ Well-defined exception types already available
- ✅ Incremental implementation approach
- ✅ Comprehensive testing strategy

### Potential Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Breaking existing functionality** | Incremental updates with testing after each phase |
| **Circular import issues** | Careful import analysis and testing |
| **Performance impact** | Exception handling is already in place, minimal impact |
| **Inconsistent error messages** | Standardized error message templates |

## Success Criteria

### Functional Requirements
- ✅ All generic `except Exception as e:` patterns replaced with specific exceptions
- ✅ No existing functionality broken
- ✅ All imports correctly updated
- ✅ Exception chaining preserved with `from e`

### Quality Requirements  
- ✅ Consistent error reporting across all services
- ✅ Proper integration with centralized error handling
- ✅ Meaningful error messages with context
- ✅ Maintained graceful degradation where appropriate

### Technical Requirements
- ✅ All tests pass
- ✅ No circular import issues
- ✅ Proper exception hierarchy usage
- ✅ Error handlers correctly catch new exception types

## Implementation Checklist

### Phase 1: Infrastructure ✅
- [ ] Add new exception types to `app/exceptions.py`
- [ ] Add new error handlers to `app/error_handlers.py`
- [ ] Update `EXCEPTION_HANDLERS` mapping
- [ ] Test new exception types work correctly

### Phase 2: Core Files ✅
- [ ] Update `chroma_client.py`
- [ ] Update `ollama_embeddings.py`
- [ ] Update `prompt_manager.py`
- [ ] Update `workflow_coordinator.py`
- [ ] Test core functionality

### Phase 3: Service Files ✅
- [ ] Update relevance services (8 files)
- [ ] Update core services (6 files)
- [ ] Update analysis services (4 files)
- [ ] Update graph services (10+ files)
- [ ] Update cache services (5 files)
- [ ] Update infrastructure services (4 files)
- [ ] Test service layer functionality

### Phase 4: Pipeline & Analyzers ✅
- [ ] Update pipeline files (3 files)
- [ ] Update analyzer files (2 files)
- [ ] Final integration testing
- [ ] Performance validation

### Final Validation ✅
- [ ] All tests pass
- [ ] Error handling integration verified
- [ ] Documentation updated
- [ ] Code review completed

## Conclusion

This refactoring plan addresses the Priority 1 item "Improve Error Handling Integration" from the code analysis report <mcreference link="https://www.freecodecamp.org/news/improve-and-restructure-codebase-with-ai-tools/" index="0">0</mcreference>. The implementation follows best practices for code refactoring <mcreference link="https://www.techtarget.com/searchapparchitecture/definition/refactoring" index="2">2</mcreference> <mcreference link="https://www.codesee.io/learning-center/code-refactoring" index="3">3</mcreference>, ensuring that:

1. **No functionality is broken** through incremental changes and thorough testing
2. **All dependent code is corrected** with proper import updates
3. **New implementation only** - no fallbacks, just improved error handling
4. **Proper integration** with the existing centralized error handling system

The estimated 1-2 days effort will significantly improve the maintainability, debuggability, and user experience of the Veritas application by providing consistent, meaningful error reporting across all agent services <mcreference link="https://www.anthropic.com/engineering/claude-code-best-practices" index="1">1</mcreference>.
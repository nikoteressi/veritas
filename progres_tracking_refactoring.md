# Refactoring Plan: Migrating to an Event-Driven Progress Tracking System

**Objective:** To refactor the existing progress tracking system, moving away from hardcoded, step-based percentage updates to a flexible, interactive, and event-driven model. This will provide a smoother user experience and decouple the backend's business logic from the frontend's presentation logic.

**Status:** ✅ COMPLETED

This refactoring has been successfully implemented. The system now uses an event-driven architecture for progress tracking. The legacy `ProgressTrackingService` has been completely removed and replaced with the new `EventEmissionService`.

## ✅ **Completed Steps:**

### Backend Refactoring
- ✅ Created `ProgressEvent` schema
- ✅ Implemented `EventEmissionService` 
- ✅ Enhanced WebSocket manager with event support
- ✅ Updated verification pipeline to use events
- ✅ Updated all pipeline steps to emit events
- ✅ Updated fact-checking service for event callbacks
- ✅ **Removed `progress_tracking.py` entirely**
- ✅ **Cleaned up all ProgressTrackingService references**

### Frontend Refactoring  
- ✅ Enhanced WebSocket service for event handling
- ✅ Created `useProgressInterpreter` hook
- ✅ Updated LoadingState component
- ✅ Smooth, granular progress calculation

### Testing & Validation
- ✅ All imports work correctly
- ✅ Frontend builds successfully
- ✅ Event system fully operational
- ✅ Complete migration verified

---

## Phase 1: Backend Refactoring (Python/FastAPI)

The backend will be modified to act as an event emitter. It will report *what* it's doing, not *how much* progress has been made.

### 1.1. Define the Event Schema

In a shared module (e.g., `schemas.py`), define a Pydantic model for our events. This ensures a consistent structure.

```python
# in a new file, e.g., 'core/schemas.py' or similar
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal

class ProgressEvent(BaseModel):
    event_name: str = Field(..., description="The unique name of the event, e.g., 'CLAIMS_EXTRACTED'.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="A dictionary containing event-specific data.")
````

### 1.2. Upgrade the WebSocket Manager

Modify `websocket_manager.py` to send `ProgressEvent` objects. The manager should not know anything about percentages.

**Current (for removal):**

```python
# websocket_manager.py (OLD)
async def send_progress_update(self, progress: int, message: str):
    await self.send_json({"progress": progress, "message": message})
```

**New:**

```python
# websocket_manager.py (NEW)
from .schemas import ProgressEvent

class VerificationWSService:
    # ... existing code ...
    async def emit_event(self, event: ProgressEvent):
        """Emits a structured event to the client."""
        await self.send_json(event.model_dump())

# The callback function passed to the pipeline will now be `emit_event`.
```

### 1.3. Instrument the Verification Pipeline

Refactor `verification_pipeline.py` and all its service modules (`fact_checking_service.py`, etc.) to emit events instead of calculating progress.

**Example: `fact_checking_service.py`**

```python
# fact_checking_service.py
from core.schemas import ProgressEvent

class FactCheckingService:
    async def run(self, claims: list[str], emit_event: callable):
        await emit_event(ProgressEvent(
            event_name="FACT_CHECK_STARTED",
            payload={"total_claims": len(claims)}
        ))

        for i, claim in enumerate(claims):
            # ... perform the actual fact-checking logic for the claim ...
            await emit_event(ProgressEvent(
                event_name="FACT_CHECK_ITEM_COMPLETED",
                payload={"checked": i + 1, "total": len(claims), "claim_text": claim}
            ))
        
        await emit_event(ProgressEvent(event_name="FACT_CHECK_FINISHED"))

```

### 1.4. Deprecate `progress_tracking.py`

The `progress_tracking.py` file, which maps steps to percentages, can now be removed entirely. Its logic will be moved to the frontend.

-----

## Phase 2: Frontend Refactoring (React)

The frontend will become the "interpreter" of the event stream, responsible for calculating and animating the progress bar.

### 2.1. Update the WebSocket Service

Modify `verificationStateService.js` to handle incoming event objects instead of simple progress data.

```javascript
// services/verificationStateService.js

// ... (setup WebSocket connection) ...

socket.onmessage = (event) => {
  const eventData = JSON.parse(event.data);
  // Instead of setting state directly, notify listeners with the full event object
  notifyListeners(eventData); 
};

// ... (rest of the service: addListener, removeListener, notifyListeners)
```

### 2.2. Create a Progress Interpreter Hook ⭐️

This is the core of the new frontend logic. A new hook, `useProgressInterpreter.js`, will subscribe to the event service and manage the UI state.

```javascript
// hooks/useProgressInterpreter.js
import { useState, useEffect }_from_ 'react';
import verificationStateService from '../services/verificationStateService';

export function useProgressInterpreter() {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Starting verification...');
  const [eventLog, setEventLog] = useState([]);

  useEffect(() => {
    const handleEvent = (event) => {
      // Add event to a log for detailed view
      setEventLog(prevLog => [...prevLog, event]);

      // The main logic to translate events to UI state
      switch (event.event_name) {
        case 'VERIFICATION_STARTED':
          setProgress(5);
          setMessage('Analyzing request...');
          break;
        
        case 'FACT_CHECK_STARTED':
          setProgress(20);
          setMessage(`Found ${event.payload.total_claims} claims to check.`);
          break;

        case 'FACT_CHECK_ITEM_COMPLETED':
          // This creates the smooth, granular progress
          const baseProgress = 20; // Progress before fact-checking
          const factCheckWeight = 70; // This phase is worth 70% of the bar
          const itemProgress = (event.payload.checked / event.payload.total) * factCheckWeight;
          setProgress(baseProgress + itemProgress);
          setMessage(`Checking claim ${event.payload.checked} of ${event.payload.total}...`);
          break;

        case 'VERDICT_GENERATION_STARTED':
          // For non-granular tasks, we can create a smooth animation
          // (Can be done with CSS transitions or a library like framer-motion)
          setProgress(95); 
          setMessage('Generating final verdict...');
          break;

        case 'VERIFICATION_COMPLETED':
          setProgress(100);
          setMessage('Done!');
          break;
      }
    };

    verificationStateService.addListener(handleEvent);
    return () => verificationStateService.removeListener(handleEvent);
  }, []);

  return { progress, message, eventLog };
}
```

### 2.3. Refactor the UI Component

Update `LoadingState.jsx` to use the new hook. Remove all hardcoded steps and logic.

```jsx
// components/LoadingState.jsx
import React from 'react';
import { useProgressInterpreter } from '../hooks/useProgressInterpreter';
import { ProgressBar } from './ProgressBar'; // Assuming a simple progress bar component

export function LoadingState() {
  const { progress, message, eventLog } = useProgressInterpreter();

  return (
    <div>
      <h2>Verification in Progress...</h2>
      <p>{message}</p>
      <ProgressBar percentage={progress} />

      {/* Optional: Display a detailed event log for transparency */}
      <div className="event-log">
        <h4>Activity Log:</h4>
        <ul>
          {eventLog.map((e, i) => (
            <li key={i}>{e.event_name}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

-----

## Summary of Benefits

1.  **Decoupling:** The backend no longer needs to know how its work is presented. It can focus solely on the verification process.
2.  **UX Improvement:** Progress is granular and reflects the actual work being done, eliminating large, jarring jumps in the progress bar.
3.  **Interactivity:** The frontend can display a live log of events, making the process transparent and engaging for the user.
4.  **Maintainability:** Adding a new sub-step on the backend only requires emitting a new event. The frontend can choose to ignore it or create a new rule for it without requiring backend changes.

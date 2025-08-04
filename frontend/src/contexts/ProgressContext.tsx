import React, { createContext, useContext, ReactNode, useReducer, useMemo, useCallback } from 'react'
import { ProgressState, ProgressStep, StepStatus } from '../types'

// Progress Context Interface
interface ProgressContextValue {
  // State
  progressState: ProgressState
  currentMessage: string
  sessionId: string | null
  
  // Computed values
  activeStepId: string | null
  currentStep: ProgressStep | null
  isActive: boolean
  
  // Actions
  setSteps: (steps: ProgressStep[]) => void
  updateProgress: (progress: number, message?: string) => void
  updateStep: (stepId: string, status: StepStatus, progress?: number, message?: string) => void
  resetProgress: () => void
  setSessionId: (sessionId: string | null) => void
  setCurrentMessage: (message: string) => void
}

// Progress Actions
type ProgressAction =
  | { type: 'SET_STEPS'; payload: ProgressStep[] }
  | { type: 'UPDATE_PROGRESS'; payload: { progress: number; message?: string } }
  | { type: 'UPDATE_STEP'; payload: { stepId: string; status: StepStatus; progress?: number; message?: string } }
  | { type: 'RESET_PROGRESS' }
  | { type: 'SET_SESSION_ID'; payload: string | null }
  | { type: 'SET_CURRENT_MESSAGE'; payload: string }

// Initial state
const initialProgressState: ProgressState = {
  steps: [],
  currentStepIndex: 0,
  overallProgress: 0,
  isActive: false,
}

const initialState = {
  progressState: initialProgressState,
  currentMessage: 'Ready to start verification...',
  sessionId: null as string | null,
}

// Progress Reducer
function progressReducer(state: typeof initialState, action: ProgressAction): typeof initialState {
  switch (action.type) {
    case 'SET_STEPS':
      return {
        ...state,
        progressState: {
          ...state.progressState,
          steps: action.payload,
          currentStepIndex: 0,
          isActive: action.payload.length > 0,
        },
      }
      
    case 'UPDATE_PROGRESS':
      return {
        ...state,
        progressState: {
          ...state.progressState,
          overallProgress: Math.max(0, Math.min(100, action.payload.progress)),
        },
        currentMessage: action.payload.message || state.currentMessage,
      }
      
    case 'UPDATE_STEP': {
      const { stepId, status, progress, message } = action.payload
      const updatedSteps = state.progressState.steps.map(step =>
        step.id === stepId
          ? { ...step, status, ...(progress !== undefined && { progress }) }
          : step
      )
      
      // Update current step index based on step status
      let newCurrentStepIndex = state.progressState.currentStepIndex
      const stepIndex = updatedSteps.findIndex(step => step.id === stepId)
      
      if (stepIndex !== -1) {
        if (status === 'running') {
          newCurrentStepIndex = stepIndex
        } else if (status === 'completed' && stepIndex === newCurrentStepIndex) {
          // Move to next step if current step is completed
          newCurrentStepIndex = Math.min(stepIndex + 1, updatedSteps.length - 1)
        }
      }
      
      return {
        ...state,
        progressState: {
          ...state.progressState,
          steps: updatedSteps,
          currentStepIndex: newCurrentStepIndex,
        },
        currentMessage: message || state.currentMessage,
      }
    }
    
    case 'RESET_PROGRESS':
      return {
        ...state,
        progressState: initialProgressState,
        currentMessage: 'Ready to start verification...',
      }
      
    case 'SET_SESSION_ID':
      return {
        ...state,
        sessionId: action.payload,
      }
      
    case 'SET_CURRENT_MESSAGE':
      return {
        ...state,
        currentMessage: action.payload,
      }
      
    default:
      return state
  }
}

interface ProgressProviderProps {
  children: ReactNode
}

// Create context
const ProgressContext = createContext<ProgressContextValue | null>(null)

export const useProgressContext = () => {
  const context = useContext(ProgressContext)
  if (!context) {
    throw new Error('useProgressContext must be used within a ProgressProvider')
  }
  return context
}

export const ProgressProvider: React.FC<ProgressProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(progressReducer, initialState)
  
  // Actions
  const setSteps = useCallback((steps: ProgressStep[]) => {
    dispatch({ type: 'SET_STEPS', payload: steps })
  }, [])
  
  const updateProgress = useCallback((progress: number, message?: string) => {
    dispatch({ type: 'UPDATE_PROGRESS', payload: { progress, ...(message !== undefined && { message }) } })
  }, [])
  
  const updateStep = useCallback((stepId: string, status: StepStatus, progress?: number, message?: string) => {
    dispatch({ type: 'UPDATE_STEP', payload: { 
      stepId, 
      status, 
      ...(progress !== undefined && { progress }),
      ...(message !== undefined && { message })
    } })
  }, [])
  
  const resetProgress = useCallback(() => {
    dispatch({ type: 'RESET_PROGRESS' })
  }, [])
  
  const setSessionId = useCallback((sessionId: string | null) => {
    dispatch({ type: 'SET_SESSION_ID', payload: sessionId })
  }, [])
  
  const setCurrentMessage = useCallback((message: string) => {
    dispatch({ type: 'SET_CURRENT_MESSAGE', payload: message })
  }, [])
  
  // Computed values
  const activeStepId = useMemo(() => {
    const currentStep = state.progressState.steps[state.progressState.currentStepIndex]
    return currentStep?.id || null
  }, [state.progressState.steps, state.progressState.currentStepIndex])
  
  const currentStep = useMemo(() => {
    return state.progressState.steps[state.progressState.currentStepIndex] || null
  }, [state.progressState.steps, state.progressState.currentStepIndex])
  
  const isActive = useMemo(() => {
    return state.progressState.isActive
  }, [state.progressState.isActive])
  
  // Memoized context value to prevent unnecessary re-renders
  const contextValue: ProgressContextValue = useMemo(() => ({
    // State
    progressState: state.progressState,
    currentMessage: state.currentMessage,
    sessionId: state.sessionId,
    
    // Computed values
    activeStepId,
    currentStep,
    isActive,
    
    // Actions
    setSteps,
    updateProgress,
    updateStep,
    resetProgress,
    setSessionId,
    setCurrentMessage,
  }), [
    state.progressState,
    state.currentMessage,
    state.sessionId,
    activeStepId,
    currentStep,
    isActive,
    setSteps,
    updateProgress,
    updateStep,
    resetProgress,
    setSessionId,
    setCurrentMessage,
  ])

  return (
    <ProgressContext.Provider value={contextValue}>
      {children}
    </ProgressContext.Provider>
  )
}
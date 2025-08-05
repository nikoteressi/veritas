import { useEffect, useCallback, useMemo } from 'react';
import { webSocketService } from '../services/webSocketService';
import { useProgressContext } from '../contexts/ProgressContext';
import { ProgressStep, StepStatus } from '../types';

// WebSocket message interfaces
interface StepsDefinitionData {
  steps: ProgressStep[];
}



interface StepUpdateData {
  step_id: string;
  status: StepStatus;
  progress?: number;
  message?: string;
}

/**
 * Hook for interpreting verification progress from WebSocket messages.
 * Uses ProgressContext for centralized state management.
 */
export function useProgressInterpreter() {
  // Get context
  const {
    updateStep,
    setSteps,
    setSessionId,
    resetProgress: contextResetProgress
  } = useProgressContext();
  
  const resetProgress = useCallback(() => {
    contextResetProgress();
  }, [contextResetProgress]);

  // WebSocket message handlers - memoized to prevent unnecessary re-subscriptions
  const handleStepsDefinition = useCallback((data: StepsDefinitionData) => {
    console.log('📋 Steps definition received:', data);
    setSteps(data.steps);
  }, [setSteps]);

  const handleStepUpdate = useCallback((data: StepUpdateData) => {
    console.log('👣 Step update received:', data);
    
    updateStep(data.step_id, data.status, data.progress, data.message);
  }, [updateStep]);

  const handleSessionEstablished = useCallback((data: any) => {
    console.log('🔗 Session established:', data);
    if (data.session_id) {
      setSessionId(data.session_id);
    }
  }, [setSessionId]);

  // WebSocket subscriptions - optimized dependencies
  useEffect(() => {
    const unsubscribeSteps = webSocketService.subscribe('steps_definition', handleStepsDefinition);
    const unsubscribeStepUpdate = webSocketService.subscribe('step_update', handleStepUpdate);
    const unsubscribeSession = webSocketService.subscribe('session_established', handleSessionEstablished);

    return () => {
      unsubscribeSteps();
      unsubscribeStepUpdate();
      unsubscribeSession();
    };
  }, [handleStepsDefinition, handleStepUpdate, handleSessionEstablished]);

  // Return utility functions for external use - memoized
  return useMemo(() => ({
    resetProgress,
  }), [resetProgress]);
}
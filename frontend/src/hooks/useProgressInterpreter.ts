import { useEffect, useRef, useCallback, useMemo } from 'react';
import { webSocketService } from '../services/webSocketService';
import { createProgressAnimator } from '../utils/progressAnimator';
import { useProgressContext } from '../contexts/ProgressContext';
import { ProgressStep, StepStatus } from '../types';

// WebSocket message interfaces
interface StepsDefinitionData {
  steps: ProgressStep[];
}

interface ProgressUpdateData {
  current_progress: number;
  message?: string;
  animation_duration?: number;
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
    updateProgress,
    updateStep,
    setSteps,
    setSessionId,
    setCurrentMessage,
    resetProgress: contextResetProgress
  } = useProgressContext();
  
  // Animation ref
  const animatorRef = useRef(createProgressAnimator((value: number) => {
    updateProgress(value);
  }));
  
  // Animation functions
  const animateProgress = useCallback((targetProgress: number, duration: number = 300) => {
    const animator = animatorRef.current;
    
    animator.animateTo(targetProgress, duration);
  }, []);
  
  const setProgressImmediate = useCallback((progress: number) => {
    updateProgress(Math.max(0, Math.min(100, progress)));
  }, [updateProgress]);
  
  const resetProgress = useCallback(() => {
    contextResetProgress();
  }, [contextResetProgress]);

  // WebSocket message handlers - memoized to prevent unnecessary re-subscriptions
  const handleStepsDefinition = useCallback((data: StepsDefinitionData) => {
    console.log('📋 Steps definition received:', data);
    setSteps(data.steps);
  }, [setSteps]);

  const handleProgressUpdate = useCallback((data: ProgressUpdateData) => {
    console.log('📊 Progress update received:', data);
    
    const progressValue = Math.max(0, Math.min(100, data.current_progress));
    animateProgress(progressValue, data.animation_duration || 200);
    
    if (data.message) {
      setCurrentMessage(data.message);
    }
  }, [animateProgress, setCurrentMessage]);

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
    const unsubscribeProgress = webSocketService.subscribe('progress_update', handleProgressUpdate);
    const unsubscribeStepUpdate = webSocketService.subscribe('step_update', handleStepUpdate);
    const unsubscribeSession = webSocketService.subscribe('session_established', handleSessionEstablished);

    return () => {
      unsubscribeSteps();
      unsubscribeProgress();
      unsubscribeStepUpdate();
      unsubscribeSession();
    };
  }, [handleStepsDefinition, handleProgressUpdate, handleStepUpdate, handleSessionEstablished]);

  // Return utility functions for external use - memoized
  return useMemo(() => ({
    animateProgress,
    setProgressImmediate,
    resetProgress,
  }), [animateProgress, setProgressImmediate, resetProgress]);
}
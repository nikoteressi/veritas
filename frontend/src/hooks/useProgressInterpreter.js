import { useState, useEffect, useRef, useCallback } from 'react';
import { verificationStateService } from '../services/verificationStateService';
import { webSocketService } from '../services/webSocketService';
import ProgressAnimator, { createProgressAnimator } from '../utils/progressAnimator';

/**
 * Enhanced hook for interpreting verification events and calculating progress.
 * Supports both legacy events and new dynamic WebSocket-based progress system.
 */
export function useProgressInterpreter() {
  // Core progress state
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Starting verification...');
  const [eventLog, setEventLog] = useState([]);
  const [currentStep, setCurrentStep] = useState(null);
  
  // Dynamic progress system state
  const [steps, setSteps] = useState([]);
  const [visibleSteps, setVisibleSteps] = useState([]); // Only steps that have been activated
  const [activeStepId, setActiveStepId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLegacyMode, setIsLegacyMode] = useState(true);
  
  // Animation and refs
  const progressRef = useRef(0);
  const animatorRef = useRef(null);

  // Initialize progress animator
  useEffect(() => {
    animatorRef.current = createProgressAnimator.smooth(
      (value) => {
        progressRef.current = value;
        setProgress(Math.round(value));
      },
      (finalValue) => {
        // Progress animation completed
      }
    );

    return () => {
      if (animatorRef.current) {
        animatorRef.current.stop();
      }
    };
  }, []);

  // Smooth progress animation using new animator
  const animateProgress = useCallback((targetProgress, duration = 300) => {
    if (animatorRef.current) {
      animatorRef.current.animateTo(targetProgress, duration);
    }
  }, []);

  // Set progress immediately without animation
  const setProgressImmediate = useCallback((value) => {
    if (animatorRef.current) {
      animatorRef.current.setImmediate(value);
    }
  }, []);

  // WebSocket message handlers for new progress system
  const handleStepsDefinition = useCallback((data) => {
    // Validate steps data
    if (!data || !Array.isArray(data.steps) || data.steps.length === 0) {
      return;
    }

    // Valid steps data received
    const stepsWithDefaults = data.steps.map(step => ({
      ...step,
      status: step.status || 'pending',
      progress: step.progress || 0
    }));

    setSteps(stepsWithDefaults);
    setVisibleSteps([data.steps[0].id]);
    setActiveStepId(data.steps[0].id);
  }, []);

  const handleProgressUpdate = useCallback((data) => {
    const progressValue = Math.max(0, Math.min(100, data.current_progress || 0));
    
    animateProgress(progressValue, data.animation_duration || 200);
    
    if (data.message) {
      setMessage(data.message);
    }
  }, [animateProgress]);

  const handleStepUpdate = useCallback((data) => {
    const updatedSteps = steps.map(step => 
      step.id === data.step_id 
        ? { ...step, status: data.status, progress: data.progress, message: data.message }
        : step
    );
    setSteps(updatedSteps);

    // Add step to visible steps if not already visible
    if (!visibleSteps.includes(data.step_id)) {
      setVisibleSteps(prev => [...prev, data.step_id]);
    }

    // Update active step
    if (data.status === 'in_progress') {
      setActiveStepId(data.step_id);
      const newCurrentStep = updatedSteps.find(step => step.id === data.step_id);
      if (newCurrentStep) {
        setCurrentStep(newCurrentStep);
      }
    }
  }, [steps, visibleSteps]);

  // WebSocket message handling using existing service
  useEffect(() => {
    // Get current sessionId from webSocketService
    const currentStatus = webSocketService.getStatus();
    if (currentStatus.sessionId && currentStatus.sessionId !== sessionId) {
      setSessionId(currentStatus.sessionId);
    }
    
    // Subscribe to progress-related messages
    const unsubscribeStepsDefinition = webSocketService.subscribe('steps_definition', (data) => {
      handleStepsDefinition(data);
    });

    const unsubscribeProgressUpdate = webSocketService.subscribe('progress_update', (data) => {
      handleProgressUpdate(data);
    });

    const unsubscribeStepUpdate = webSocketService.subscribe('step_update', (data) => {
      handleStepUpdate(data);
    });

    // Subscribe to legacy progress events for backward compatibility
    const unsubscribeProgressEvent = webSocketService.subscribe('progress_event', (data) => {
      handleLegacyEvent(data);
    });

    // Subscribe to session established events
    const unsubscribeSessionEstablished = webSocketService.subscribe('session_established', () => {
      const status = webSocketService.getStatus();
      setSessionId(status.sessionId);
    });

    return () => {
      unsubscribeStepsDefinition();
      unsubscribeProgressUpdate();
      unsubscribeStepUpdate();
      unsubscribeProgressEvent();
      unsubscribeSessionEstablished();
    };
  }, [handleStepsDefinition, handleProgressUpdate, handleStepUpdate, handleLegacyEvent, sessionId]);

  // Legacy event handler for backward compatibility
  const handleLegacyEvent = useCallback((eventData) => {
    // Add event to log for detailed view
    setEventLog(prevLog => [...prevLog, eventData]);

    // Only process legacy events if not in new progress mode
    if (!isLegacyMode) return;

    // Translate events to UI state with smooth progress
    switch (eventData.event_name) {
      case 'VERIFICATION_STARTED':
        animateProgress(5, 500);
        setMessage('Analyzing request...');
        setCurrentStep('validation');
        break;
      
      case 'SCREENSHOT_PARSING_STARTED':
        animateProgress(25, 800);
        setMessage('Parsing screenshot...');
        setCurrentStep('screenshot_parsing');
        break;

      case 'SCREENSHOT_PARSING_COMPLETED':
        animateProgress(35, 400);
        setMessage('Screenshot parsing complete.');
        break;

      case 'POST_ANALYSIS_STARTED':
      case 'TEMPORAL_ANALYSIS_STARTED':
      case 'REPUTATION_RETRIEVAL_STARTED':
        animateProgress(45, 800);
        setMessage('Analyzing context...');
        setCurrentStep('analyzing_context');
        break;

      case 'POST_ANALYSIS_COMPLETED':
      case 'TEMPORAL_ANALYSIS_COMPLETED':
      case 'REPUTATION_RETRIEVAL_COMPLETED':
        animateProgress(55, 400);
        setMessage('Context analysis complete.');
        break;
      
      case 'FACT_CHECKING_STARTED':
        const totalClaims = eventData.payload?.total_claims || 0;
        animateProgress(65, 500);
        setMessage(`Found ${totalClaims} claims to fact-check`);
        setCurrentStep('fact_checking');
        break;

      case 'FACT_CHECKING_ITEM_COMPLETED':
        // Smooth granular progress during fact-checking
        const baseProgress = 65;
        const factCheckWeight = 20; // 65% to 85%
        const checked = eventData.payload?.checked || 0;
        const total = eventData.payload?.total || 1;
        const itemProgress = (checked / total) * factCheckWeight;
        const targetProgress = baseProgress + itemProgress;
        
        // Smooth animation for each item
        animateProgress(targetProgress, 200);
        setMessage(`Checking claim ${checked} of ${total}...`);
        break;

      case 'FACT_CHECKING_COMPLETED':
        animateProgress(85, 600);
        setMessage('Fact-checking complete');
        break;

      case 'SUMMARIZATION_STARTED':
      case 'VERDICT_GENERATION_STARTED':
      case 'MOTIVES_ANALYSIS_STARTED':
        animateProgress(90, 600);
        setMessage('Summarizing results...');
        setCurrentStep('summarization');
        break;

      case 'SUMMARIZATION_COMPLETED':
      case 'VERDICT_GENERATION_COMPLETED':
      case 'MOTIVES_ANALYSIS_COMPLETED':
        animateProgress(95, 400);
        setMessage('Summarization complete.');
        break;

      case 'REPUTATION_UPDATE_STARTED':
        animateProgress(96, 200);
        setMessage('Updating user reputation...');
        setCurrentStep('reputation_update');
        break;

      case 'REPUTATION_UPDATE_COMPLETED':
        animateProgress(97, 200);
        setMessage('User reputation updated');
        break;

      case 'RESULT_STORAGE_STARTED':
        animateProgress(98, 200);
        setMessage('Saving results...');
        setCurrentStep('result_storage');
        break;

      case 'RESULT_STORAGE_COMPLETED':
        animateProgress(99, 200);
        setMessage('Results saved successfully');
        break;

      case 'VERIFICATION_COMPLETED':
        animateProgress(100, 300);
        setMessage('Verification complete!');
        setCurrentStep('completion');
        break;

      default:
        // Unknown event - just log it
        console.log('Unknown progress event:', eventData.event_name);
    }
  }, [isLegacyMode, animateProgress]);

  // Define verification steps with smooth progress ranges (legacy)
  const verificationSteps = [
    { key: 'validation', label: 'Validation', progress: 5, duration: 500 },
    { key: 'screenshot_parsing', label: 'Screenshot Parsing', progress: 15, duration: 800 },
    { key: 'temporal_analysis', label: 'Temporal Analysis', progress: 25, duration: 700 },
    { key: 'post_analysis', label: 'Post Analysis', progress: 35, duration: 800 },
    { key: 'reputation_retrieval', label: 'Reputation Check', progress: 45, duration: 600 },
    { key: 'fact_checking', label: 'Fact Checking', progress: 80, duration: 2000 },
    { key: 'summarization', label: 'Summarization', progress: 85, duration: 600 },
    { key: 'verdict_generation', label: 'Verdict Generation', progress: 90, duration: 800 },
    { key: 'motives_analysis', label: 'Motives Analysis', progress: 95, duration: 500 },
    { key: 'reputation_update', label: 'Reputation Update', progress: 97, duration: 300 },
    { key: 'result_storage', label: 'Result Storage', progress: 99, duration: 300 },
    { key: 'completion', label: 'Completion', progress: 100, duration: 200 }
  ];

  const getCurrentStep = () => {
    if (!isLegacyMode && currentStep) {
      return currentStep;
    }
    
    const currentProgress = progressRef.current;
    for (let i = verificationSteps.length - 1; i >= 0; i--) {
      if (currentProgress >= verificationSteps[i].progress - 5) {
        return verificationSteps[i];
      }
    }
    return verificationSteps[0];
  };

  // Legacy verification state service integration
  useEffect(() => {
    // Subscribe to progress events
    const unsubscribe = verificationStateService.subscribe((state, previousState) => {
      // This will be called on state changes, but we're specifically interested in events
      // We'll handle events through the direct event system
    });

    // Add event listener for progress events
    const originalEmit = verificationStateService._emit;
    verificationStateService._emit = function(eventType, data) {
      if (eventType === 'progress_event') {
        handleLegacyEvent(data);
      }
      return originalEmit.call(this, eventType, data);
    };

    return () => {
      unsubscribe();
      // Restore original emit function
      verificationStateService._emit = originalEmit;
    };
  }, [handleLegacyEvent]);

  // Add comprehensive logging for debugging
  console.log('useProgressInterpreter state:', {
    progress,
    message,
    currentStep,
    isLegacyMode,
    activeStepId,
    stepsCount: steps.length,
    eventLogCount: eventLog.length,
    sessionId
  });

  return { 
    // Core progress data
    progress, 
    message, 
    eventLog, 
    currentStep,
    
    // Legacy compatibility
    verificationSteps,
    getCurrentStep,
    
    // New dynamic progress system
    steps: visibleSteps, // Return only visible steps for UI
    allSteps: steps, // Keep all steps available if needed
    activeStepId,
    sessionId,
    isLegacyMode,
    
    // Utility functions
    animateProgress,
    setProgressImmediate
  };
}
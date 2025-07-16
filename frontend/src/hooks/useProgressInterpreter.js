import { useState, useEffect, useRef } from 'react';
import { verificationStateService } from '../services/verificationStateService';

/**
 * Hook for interpreting verification events and calculating progress.
 * This hook converts event-driven progress updates into UI-friendly progress data.
 */
export function useProgressInterpreter() {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Starting verification...');
  const [eventLog, setEventLog] = useState([]);
  const [currentStep, setCurrentStep] = useState(null);
  const progressRef = useRef(0);
  const animationRef = useRef(null);

  // Smooth progress animation
  const animateProgress = (targetProgress, duration = 1000) => {
    const startProgress = progressRef.current;
    const startTime = Date.now();
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function for smooth animation
      const easeProgress = progress * progress * (3 - 2 * progress);
      const currentProgress = startProgress + (targetProgress - startProgress) * easeProgress;
      
      progressRef.current = currentProgress;
      setProgress(Math.round(currentProgress));
      
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    animationRef.current = requestAnimationFrame(animate);
  };

  // Define verification steps with smooth progress ranges
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
    const currentProgress = progressRef.current;
    for (let i = verificationSteps.length - 1; i >= 0; i--) {
      if (currentProgress >= verificationSteps[i].progress - 5) {
        return verificationSteps[i];
      }
    }
    return verificationSteps[0];
  };

  useEffect(() => {
    const handleEvent = (eventData) => {
      // Add event to log for detailed view
      setEventLog(prevLog => [...prevLog, eventData]);

      // Translate events to UI state with smooth progress
      switch (eventData.event_name) {
        case 'VERIFICATION_STARTED':
          animateProgress(5, 500);
          setMessage('Analyzing request...');
          setCurrentStep('validation');
          break;
        
        case 'SCREENSHOT_PARSING_STARTED':
          animateProgress(15, 800);
          setMessage('Parsing screenshot...');
          setCurrentStep('screenshot_parsing');
          break;

        case 'SCREENSHOT_PARSING_COMPLETED':
          animateProgress(22, 400);
          setMessage('Screenshot parsing complete.');
          break;

        case 'POST_ANALYSIS_STARTED':
          animateProgress(35, 800);
          setMessage('Analyzing post...');
          setCurrentStep('post_analysis');
          break;

        case 'POST_ANALYSIS_COMPLETED':
          animateProgress(42, 400);
          setMessage('Post analysis complete.');
          break;
        
        case 'REPUTATION_RETRIEVAL_STARTED':
          animateProgress(25, 600);
          setMessage('Retrieving user reputation...');
          setCurrentStep('reputation_retrieval');
          break;
        
        case 'REPUTATION_RETRIEVAL_COMPLETED':
          const username = eventData.payload?.username || 'unknown';
          animateProgress(32, 400);
          setMessage(`Retrieved reputation for ${username}`);
          break;
        
        case 'TEMPORAL_ANALYSIS_STARTED':
          animateProgress(35, 700);
          setMessage('Performing temporal analysis...');
          setCurrentStep('temporal_analysis');
          break;
        
        case 'TEMPORAL_ANALYSIS_COMPLETED':
          animateProgress(42, 400);
          setMessage('Temporal analysis complete');
          break;
        
        case 'FACT_CHECKING_STARTED':
          const totalClaims = eventData.payload?.total_claims || 0;
          animateProgress(45, 500);
          setMessage(`Found ${totalClaims} claims to fact-check`);
          setCurrentStep('fact_checking');
          break;

        case 'FACT_CHECKING_ITEM_COMPLETED':
          // Smooth granular progress during fact-checking
          const baseProgress = 45;
          const factCheckWeight = 35; // 45% to 80%
          const checked = eventData.payload?.checked || 0;
          const total = eventData.payload?.total || 1;
          const itemProgress = (checked / total) * factCheckWeight;
          const targetProgress = baseProgress + itemProgress;
          
          // Smooth animation for each item
          animateProgress(targetProgress, 200);
          setMessage(`Checking claim ${checked} of ${total}...`);
          break;

        case 'FACT_CHECKING_COMPLETED':
          animateProgress(80, 600);
          setMessage('Fact-checking complete');
          break;

        case 'SUMMARIZATION_STARTED':
          animateProgress(85, 600);
          setMessage('Summarizing results...');
          setCurrentStep('summarization');
          break;

        case 'SUMMARIZATION_COMPLETED':
          animateProgress(88, 400);
          setMessage('Summarization complete.');
          break;

        case 'VERDICT_GENERATION_STARTED':
          animateProgress(85, 500);
          setMessage('Generating final verdict...');
          setCurrentStep('verdict_generation');
          break;

        case 'VERDICT_GENERATION_COMPLETED':
          const verdict = eventData.payload?.verdict || 'unknown';
          animateProgress(90, 400);
          setMessage(`Verdict: ${verdict}`);
          break;

        case 'MOTIVES_ANALYSIS_STARTED':
          animateProgress(92, 300);
          setMessage('Analyzing potential motives...');
          setCurrentStep('motives_analysis');
          break;

        case 'MOTIVES_ANALYSIS_COMPLETED':
          animateProgress(95, 300);
          setMessage('Motives analysis complete');
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
    };

    // Subscribe to progress events
    const unsubscribe = verificationStateService.subscribe((state, previousState) => {
      // This will be called on state changes, but we're specifically interested in events
      // We'll handle events through the direct event system
    });

    // Add event listener for progress events
    const originalEmit = verificationStateService._emit;
    verificationStateService._emit = function(eventType, data) {
      if (eventType === 'progress_event') {
        handleEvent(data);
      }
      return originalEmit.call(this, eventType, data);
    };

    return () => {
      unsubscribe();
      // Restore original emit function
      verificationStateService._emit = originalEmit;
      // Clean up animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return { 
    progress, 
    message, 
    eventLog, 
    currentStep,
    verificationSteps,
    getCurrentStep: getCurrentStep
  };
}
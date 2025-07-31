import React from 'react'
import { useProgressInterpreter } from '../../hooks/useProgressInterpreter'
import DynamicProgressSteps from '../DynamicProgressSteps'

function LoadingState() {
  const { 
    progress, 
    currentStep, 
    steps, 
    activeStepId, 
    sessionId, 
    isLegacyMode 
  } = useProgressInterpreter();
  
  // Always use the new dynamic progress system
  return (
    <div className="max-w-2xl mx-auto">
      <DynamicProgressSteps 
        steps={steps || []}
        activeStepId={activeStepId}
        progress={progress}
        sessionId={sessionId}
      />
    </div>
  );
}

export default LoadingState
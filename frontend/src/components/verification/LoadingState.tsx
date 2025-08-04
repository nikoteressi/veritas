import React, { useMemo } from 'react';
import { useProgressContext } from '../../contexts/ProgressContext';
import DynamicProgressSteps from '../DynamicProgressSteps';

const LoadingState: React.FC = () => {
  const {
    progressState,
    activeStepId
  } = useProgressContext();
  
  // Memoize props to prevent unnecessary re-renders of DynamicProgressSteps
  const stepsProps = useMemo(() => ({
    steps: progressState.steps || [],
    activeStepId,
    progress: progressState.overallProgress
  }), [progressState.steps, activeStepId, progressState.overallProgress]);
  
  // Always use the new dynamic progress system
  return (
    <div className="max-w-2xl mx-auto">
      <DynamicProgressSteps {...stepsProps} />
    </div>
  );
};

export default React.memo(LoadingState);
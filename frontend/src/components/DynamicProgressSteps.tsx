import React, { useMemo } from 'react';
import { ClockIcon } from '@heroicons/react/24/solid';
import { useTranslation } from 'react-i18next';
import { ProgressStep } from '../types';
import ProgressStepItem from './ProgressStepItem';

// Define enriched step interface
interface EnrichedStep extends ProgressStep {
  isActive: boolean;
  isCompleted: boolean;
  isFailed: boolean;
  isPending: boolean;
  isInProgress: boolean;
  stepProgress: number;
  index: number;
  shouldBeVisible: boolean;
}

interface DynamicProgressStepsProps {
  steps?: ProgressStep[];
  activeStepId?: string | null;
  progress?: number;
  className?: string;
  showSubsteps?: boolean;
  compact?: boolean;
}

/**
 * Dynamic Progress Steps Component
 * 
 * Renders verification steps dynamically based on WebSocket data
 * with support for substeps, time estimates, and smooth animations.
 * Optimized for performance with memoization and component splitting.
 */
const DynamicProgressSteps: React.FC<DynamicProgressStepsProps> = ({ 
  steps = [], 
  activeStepId = null, 
  progress = 0,
  className = '',
  showSubsteps = true,
  compact = false
}) => {
  const { t } = useTranslation();

  // Calculate step status and progress with dynamic visibility
  const enrichedSteps = useMemo((): EnrichedStep[] => {
    // Find current active step index
    const activeStepIndex = steps.findIndex(step => step.id === activeStepId);
    
    return steps.map((step, index) => {
      const isActive = step.id === activeStepId;
      const isCompleted = step.status === 'completed';
      const isFailed = step.status === 'failed';
      const isPending = step.status === 'pending';
      const isInProgress = step.status === 'in_progress';
      const shouldBeVisible = activeStepIndex === -1 || index <= activeStepIndex;
      
      // Calculate individual step progress
      let stepProgress = 0;
      if (isCompleted) {
        stepProgress = 100;
      } else if ((isInProgress || isActive) && step.progress !== undefined) {
        stepProgress = step.progress;
      } else if (isActive && step.progress === undefined) {
        stepProgress = 5;
      }
      
      return {
        ...step,
        isActive,
        isCompleted,
        isFailed,
        isPending,
        isInProgress,
        stepProgress,
        index,
        shouldBeVisible
      };
    }).filter(step => step.shouldBeVisible);
  }, [steps, activeStepId]);



  // Memoized empty state
  const emptyState = useMemo(() => (
    <div className={`text-center py-8 text-gray-500 ${className}`}>
      <ClockIcon className="w-8 h-8 mx-auto mb-2 opacity-50" aria-hidden="true" />
      <p>{t('loading.waiting')}</p>
    </div>
  ), [className, t]);

  if (!steps || steps.length === 0) {
    return emptyState;
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Steps List */}
      <div className="space-y-3">
        {enrichedSteps.map((step, index) => {
          const isLastStep = index === enrichedSteps.length - 1;
          
          return (
            <ProgressStepItem
              key={step.id}
              step={step}
              isActive={step.isActive}
              isCompleted={step.isCompleted}
              isFailed={step.isFailed}
              isPending={step.isPending}
              isInProgress={step.isInProgress}
              stepProgress={step.stepProgress}
              index={step.index}
              isLastStep={isLastStep}
              compact={compact}
              showSubsteps={showSubsteps}
            />
          );
        })}
      </div>
      
      {/* Debug Information (Development Only) */}
      {import.meta.env.DEV && (
        <details className="mt-6 p-3 bg-gray-100 rounded text-xs">
          <summary className="cursor-pointer font-medium">Debug Info</summary>
          <pre className="mt-2 overflow-auto">
            {JSON.stringify({ 
              activeStepId, 
              progress, 
              stepCount: steps.length 
            }, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};

export default React.memo(DynamicProgressSteps);
import React, { useMemo } from 'react';
import { CheckCircleIcon, ClockIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';
import { CheckCircleIcon as CheckCircleIconSolid } from '@heroicons/react/24/solid';

/**
 * Dynamic Progress Steps Component
 * 
 * Renders verification steps dynamically based on WebSocket data
 * with support for substeps, time estimates, and smooth animations.
 */
const DynamicProgressSteps = ({ 
  steps = [], 
  activeStepId = null, 
  progress = 0,
  className = '',
  showSubsteps = true,
  compact = false
}) => {
  // Debug logging removed for cleaner logs
  // Calculate step status and progress
  const enrichedSteps = useMemo(() => {
    return steps.map((step, index) => {
      const isActive = step.id === activeStepId;
      const isCompleted = step.status === 'COMPLETED';
      const isFailed = step.status === 'FAILED';
      const isPending = step.status === 'PENDING';
      const isInProgress = step.status === 'IN_PROGRESS';
      
      // Calculate individual step progress
      let stepProgress = 0;
      if (isCompleted) {
        stepProgress = 100;
      } else if (isInProgress && step.progress_percentage !== undefined) {
        stepProgress = step.progress_percentage;
      }
      
      return {
        ...step,
        isActive,
        isCompleted,
        isFailed,
        isPending,
        isInProgress,
        stepProgress,
        index
      };
    });
  }, [steps, activeStepId]);

  // Get step icon based on status
  const getStepIcon = (step) => {
    if (step.isFailed) {
      return <ExclamationCircleIcon className="w-5 h-5 text-red-500" />;
    } else if (step.isCompleted) {
      return <CheckCircleIconSolid className="w-5 h-5 text-green-500" />;
    } else if (step.isInProgress) {
      return (
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      );
    } else {
      return <CheckCircleIcon className="w-5 h-5 text-gray-300" />;
    }
  };

  // Get step status color classes
  const getStepColorClasses = (step) => {
    if (step.isFailed) {
      return {
        text: 'text-red-700',
        bg: 'bg-red-50',
        border: 'border-red-200'
      };
    } else if (step.isCompleted) {
      return {
        text: 'text-green-700',
        bg: 'bg-green-50',
        border: 'border-green-200'
      };
    } else if (step.isInProgress) {
      return {
        text: 'text-blue-700',
        bg: 'bg-blue-50',
        border: 'border-blue-200'
      };
    } else {
      return {
        text: 'text-gray-500',
        bg: 'bg-gray-50',
        border: 'border-gray-200'
      };
    }
  };

  if (!steps || steps.length === 0) {
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <ClockIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>Waiting for verification steps...</p>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Overall Progress Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Verification Progress
          </h3>
          <p className="text-sm text-gray-600">
            {Math.round(progress)}% complete
          </p>
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>

      {/* Steps List */}
      <div className="space-y-3">
        {enrichedSteps.map((step, index) => {
          const colors = getStepColorClasses(step);
          const isLastStep = index === enrichedSteps.length - 1;
          
          return (
            <div key={step.id} className="relative">
              {/* Step Container */}
              <div 
                className={`
                  relative p-4 rounded-lg border transition-all duration-300
                  ${colors.bg} ${colors.border}
                  ${step.isActive ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
                  ${compact ? 'p-3' : 'p-4'}
                `}
              >
                <div className="flex items-start space-x-3">
                  {/* Step Icon */}
                  <div className="flex-shrink-0 mt-0.5">
                    {getStepIcon(step)}
                  </div>
                  
                  {/* Step Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className={`font-medium ${colors.text}`}>
                        {step.name}
                      </h4>
                    </div>
                    
                    {step.description && (
                      <p className="text-sm text-gray-600 mt-1">
                        {step.description}
                      </p>
                    )}
                    
                    {/* Step Progress Bar */}
                    {step.isInProgress && step.stepProgress > 0 && (
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${step.stepProgress}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                          {Math.round(step.stepProgress)}% complete
                        </p>
                      </div>
                    )}
                    
                    {/* Substeps */}
                    {showSubsteps && step.substeps && step.substeps.length > 0 && (
                      <div className="mt-3 space-y-1">
                        {step.substeps.map((substep, substepIndex) => (
                          <div 
                            key={substepIndex}
                            className="flex items-center space-x-2 text-sm"
                          >
                            <div className="w-3 h-3 flex-shrink-0">
                              {substep.completed ? (
                                <CheckCircleIconSolid className="w-3 h-3 text-green-500" />
                              ) : substep.active ? (
                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse ml-0.5 mt-0.5" />
                              ) : (
                                <div className="w-2 h-2 bg-gray-300 rounded-full ml-0.5 mt-0.5" />
                              )}
                            </div>
                            <span className={`
                              ${substep.completed ? 'text-green-600' : 
                                substep.active ? 'text-blue-600' : 'text-gray-500'}
                            `}>
                              {substep.name}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {/* Step Metadata */}
                    {step.metadata && Object.keys(step.metadata).length > 0 && (
                      <div className="mt-2 text-xs text-gray-500">
                        {Object.entries(step.metadata).map(([key, value]) => (
                          <span key={key} className="mr-3">
                            {key}: {String(value)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Connection Line to Next Step */}
              {!isLastStep && (
                <div className="absolute left-6 top-full w-0.5 h-3 bg-gray-300 transform -translate-x-0.5" />
              )}
            </div>
          );
        })}
      </div>
      
      {/* Debug Information (Development Only) */}
      {process.env.NODE_ENV === 'development' && (
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

export default DynamicProgressSteps;
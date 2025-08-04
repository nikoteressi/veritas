import React, { memo, useMemo } from 'react';
import { CheckCircleIcon as CheckCircleIconSolid, ExclamationCircleIcon } from '@heroicons/react/24/solid';
import { CheckCircleIcon } from '@heroicons/react/24/outline';
import { useTranslation } from 'react-i18next';
import { ProgressStep } from '../types';

interface ProgressStepItemProps {
  step: ProgressStep;
  isActive: boolean;
  isCompleted: boolean;
  isFailed: boolean;
  isPending: boolean;
  isInProgress: boolean;
  stepProgress: number;
  index: number;
  isLastStep: boolean;
  compact?: boolean;
  showSubsteps?: boolean;
}

interface ColorClasses {
  text: string;
  bg: string;
  border: string;
}

/**
 * Optimized individual progress step component
 * Memoized to prevent unnecessary re-renders
 */
const ProgressStepItem: React.FC<ProgressStepItemProps> = ({
  step,
  isActive,
  isCompleted,
  isFailed,
  isInProgress,
  stepProgress,
  isLastStep,
  compact = false,
  showSubsteps = true
}) => {
  const { t } = useTranslation();

  // Memoized step icon
  const stepIcon = useMemo(() => {
    if (isFailed) {
      return <ExclamationCircleIcon className="w-5 h-5 text-red-500" />;
    } else if (isCompleted) {
      return <CheckCircleIconSolid className="w-5 h-5 text-green-500" />;
    } else if (isInProgress || isActive) {
      return (
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      );
    } else {
      return <CheckCircleIcon className="w-5 h-5 text-gray-300" />;
    }
  }, [isFailed, isCompleted, isInProgress, isActive]);

  // Memoized color classes
  const colorClasses: ColorClasses = useMemo(() => {
    if (isFailed) {
      return {
        text: 'text-red-700',
        bg: 'bg-red-50',
        border: 'border-red-200'
      };
    } else if (isCompleted) {
      return {
        text: 'text-green-700',
        bg: 'bg-green-50',
        border: 'border-green-200'
      };
    } else if (isInProgress) {
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
  }, [isFailed, isCompleted, isInProgress]);

  // Memoized substeps rendering
  const substepsContent = useMemo(() => {
    if (!showSubsteps || !step.substeps || step.substeps.length === 0) {
      return null;
    }

    return (
      <div className="mt-3 space-y-1">
        {step.substeps.map((substep) => (
          <div 
            key={substep.id}
            className="flex items-center space-x-2 text-sm"
          >
            <div className="w-3 h-3 flex-shrink-0">
              {substep.status === 'completed' ? (
                <CheckCircleIconSolid className="w-3 h-3 text-green-500" />
              ) : substep.status === 'running' ? (
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse ml-0.5 mt-0.5" />
              ) : (
                <div className="w-2 h-2 bg-gray-300 rounded-full ml-0.5 mt-0.5" />
              )}
            </div>
            <span className={`
              ${substep.status === 'completed' ? 'text-green-600' : 
                substep.status === 'running' ? 'text-blue-600' : 'text-gray-500'}
            `}>
              {substep.title}
            </span>
          </div>
        ))}
      </div>
    );
  }, [showSubsteps, step.substeps]);

  // Memoized metadata rendering
  const metadataContent = useMemo(() => {
    if (!step.metadata || Object.keys(step.metadata).length === 0) {
      return null;
    }

    return (
      <div className="mt-2 text-xs text-gray-500">
        {Object.entries(step.metadata).map(([key, value]) => (
          <span key={key} className="mr-3">
            {key}: {String(value)}
          </span>
        ))}
      </div>
    );
  }, [step.metadata]);

  // Memoized progress bar
  const progressBar = useMemo(() => {
    if (!isInProgress || stepProgress <= 0) {
      return null;
    }

    return (
      <div className="mt-2">
        <div className="w-full bg-gray-200 rounded-full h-1.5">
          <div 
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${stepProgress}%` }}
          />
        </div>
        <p className="text-xs text-gray-500 mt-1">
          {t('loading.stepProgress', { progress: Math.round(stepProgress) })}
        </p>
      </div>
    );
  }, [isInProgress, stepProgress, t]);

  return (
    <div className="relative">
      {/* Step Container */}
      <div 
        className={`
          relative rounded-lg border transition-all duration-300
          ${colorClasses.bg} ${colorClasses.border}
          ${isActive ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
          ${compact ? 'p-3' : 'p-4'}
        `}
      >
        <div className="flex items-start space-x-3">
          {/* Step Icon */}
          <div className="flex-shrink-0 mt-0.5">
            {stepIcon}
          </div>
          
          {/* Step Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <h4 className={`font-medium ${colorClasses.text}`}>
                {step.title}
              </h4>
            </div>
            
            {step.description && (
              <p className="text-sm text-gray-600 mt-1">
                {step.description}
              </p>
            )}
            
            {/* Step Progress Bar */}
            {progressBar}
            
            {/* Substeps */}
            {substepsContent}
            
            {/* Step Metadata */}
            {metadataContent}
          </div>
        </div>
      </div>
      
      {/* Connection Line to Next Step */}
      {!isLastStep && (
        <div className="absolute left-6 top-full w-0.5 h-3 bg-gray-300 transform -translate-x-0.5" />
      )}
    </div>
  );
};

export default memo(ProgressStepItem);
import React from 'react'
import { useProgressInterpreter } from '../../hooks/useProgressInterpreter'

function LoadingState() {
  const { progress, currentStep } = useProgressInterpreter();
  
  // Define steps exactly as specified in the refactoring plan
  const progressSteps = [
    { key: 'validation', label: 'Validating request', progress: 20 },
    { key: 'screenshot_parsing', label: 'Parsing screenshot', progress: 40 },
    { key: 'analyzing_context', label: 'Analyzing context', progress: 60 },
    { key: 'fact_checking', label: 'Fact-checking with external sources', progress: 80 },
    { key: 'summarization', label: 'Summarizing results', progress: 100 }
  ];
  
  return (
    <div className="max-w-2xl mx-auto">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${Math.min(progress, 100)}%` }}
          ></div>
        </div>
        <div className="text-center mt-2 text-sm text-gray-600">
          {Math.round(progress)}% complete
        </div>
      </div>

      {/* Progress Steps */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <div className="space-y-6">
          {progressSteps.map((step, index) => {
            const isComplete = progress > step.progress;
            const isActive = currentStep === step.key || (progress >= step.progress - 20 && progress <= step.progress);
            const isUpcoming = progress < step.progress - 20;
            
            return (
              <ProgressStep
                key={step.key}
                text={step.label}
                isActive={isActive}
                isComplete={isComplete}
                isUpcoming={isUpcoming}
              />
            );
          })}
        </div>
      </div>
    </div>
  )
}

function ProgressStep({ text, isActive, isComplete, isUpcoming }) {
  return (
    <div className="flex items-center space-x-4">
      {/* Step Icon */}
      <div className="flex-shrink-0">
        {isComplete ? (
          // Checkmark for completed steps (green)
          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
        ) : isActive ? (
          // Blue indicator for current step
          <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          </div>
        ) : (
          // Gray circle for upcoming steps
          <div className="w-6 h-6 bg-gray-300 rounded-full"></div>
        )}
      </div>
      
      {/* Step Text */}
      <div className="flex-1">
        <div className={`text-base font-medium ${
          isComplete ? 'text-green-600' :
          isActive ? 'text-blue-600' :
          'text-gray-400'
        }`}>
          {text}
        </div>
        {isActive && (
          <div className="text-sm text-blue-500 mt-1 font-medium">
            In progress...
          </div>
        )}
      </div>
    </div>
  )
}

export default LoadingState
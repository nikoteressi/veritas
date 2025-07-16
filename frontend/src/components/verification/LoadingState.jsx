import React from 'react'
import { useProgressInterpreter } from '../../hooks/useProgressInterpreter'

function LoadingState() {
  const { progress, message, eventLog, currentStep, verificationSteps, getCurrentStep } = useProgressInterpreter();
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Verification Results
      </h2>

      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">
          {message}
        </span>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>{message}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Dynamic Progress Steps */}
      <div className="space-y-3 text-sm text-gray-600">
        {verificationSteps.map((step, index) => {
          const isActive = currentStep === step.key;
          const isComplete = progress > step.progress + 2; // Add small buffer for smooth transitions
          const isUpcoming = progress < step.progress - 2;
          
          return (
            <ProgressStep
              key={step.key}
              text={getStepMessage(step)}
              isActive={isActive}
              isComplete={isComplete}
              isUpcoming={isUpcoming}
            />
          );
        })}
      </div>
    </div>
  )
}

function getStepMessage(step) {
  const messages = {
    'validation': 'Validating request...',
    'screenshot_parsing': 'Parsing screenshot...',
    'temporal_analysis': 'Analyzing temporal context...',
    'post_analysis': 'Analyzing post...',
    'reputation_retrieval': 'Retrieving user reputation...',
    'fact_checking': 'Fact-checking with external sources...',
    'summarization': 'Summarizing results...',
    'verdict_generation': 'Generating final verdict...',
    'motives_analysis': 'Analyzing potential motives...',
    'reputation_update': 'Updating user reputation...',
    'result_storage': 'Saving results...',
    'completion': 'Verification complete!'
  };
  return messages[step.key] || step.label;
}

function ProgressStep({ text, isActive, isComplete, isUpcoming }) {
  return (
    <div className="flex items-center">
      <div className={`w-2 h-2 rounded-full mr-3 transition-colors duration-300 ${
        isComplete ? 'bg-green-500' :
        isActive ? 'bg-blue-500 animate-pulse' :
        isUpcoming ? 'bg-gray-300' :
        'bg-gray-400'
      }`}></div>
      <span className={`transition-colors duration-300 ${
        isComplete ? 'text-green-600' :
        isActive ? 'text-blue-600 font-medium' :
        isUpcoming ? 'text-gray-400' :
        'text-gray-600'
      }`}>
        {text}
      </span>
      {isComplete && (
        <svg className="w-4 h-4 ml-2 text-green-500 transition-opacity duration-300" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      )}
      {isActive && (
        <div className="ml-2">
          <div className="animate-spin rounded-full h-3 w-3 border-b border-blue-500"></div>
        </div>
      )}
    </div>
  )
}

export default LoadingState
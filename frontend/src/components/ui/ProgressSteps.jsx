import React from 'react'

const ProgressSteps = ({ steps, currentStep }) => {
  return (
    <div className="max-w-md mx-auto">
      <div className="space-y-4">
        {steps.map((step, index) => {
          const stepNumber = index + 1
          const isCompleted = stepNumber < currentStep
          const isCurrent = stepNumber === currentStep
          const isPending = stepNumber > currentStep
          
          return (
            <div key={step.id} className="flex items-center space-x-4">
              {/* Step Circle */}
              <div className="flex-shrink-0">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                    isCompleted
                      ? 'bg-blue-600 border-blue-600 text-white'
                      : isCurrent
                      ? 'bg-blue-100 border-blue-600 text-blue-600'
                      : 'bg-gray-100 border-gray-300 text-gray-400'
                  }`}
                >
                  {isCompleted ? (
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : (
                    <span className="text-sm font-medium">{stepNumber}</span>
                  )}
                </div>
              </div>
              
              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <div
                  className={`text-sm font-medium ${
                    isCompleted || isCurrent ? 'text-gray-900' : 'text-gray-500'
                  }`}
                >
                  {step.title}
                </div>
                {step.description && (
                  <div className="text-sm text-gray-500 mt-1">
                    {step.description}
                  </div>
                )}
                {isCurrent && step.status && (
                  <div className="text-xs text-blue-600 mt-1 font-medium">
                    {step.status}
                  </div>
                )}
              </div>
              
              {/* Loading Spinner for Current Step */}
              {isCurrent && (
                <div className="flex-shrink-0">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ProgressSteps
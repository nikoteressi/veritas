import React from 'react'

function LoadingState({ progressData }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Verification Results
      </h2>

      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">
          {progressData?.step || 'Analyzing post...'}
        </span>
      </div>

      {/* Progress Bar */}
      {progressData && (
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>{progressData.step}</span>
            <span>{progressData.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progressData.progress}%` }}
            ></div>
          </div>
          {progressData.details && (
            <p className="text-sm text-gray-500 mt-2">{progressData.details}</p>
          )}
        </div>
      )}

      <div className="space-y-3 text-sm text-gray-600">
        <ProgressStep
          text="Analyzing image content..."
          isActive={!progressData || progressData.progress < 25}
          isComplete={progressData && progressData.progress >= 25}
        />
        <ProgressStep
          text="Extracting claims and user information..."
          isActive={progressData && progressData.progress >= 25 && progressData.progress < 35}
          isComplete={progressData && progressData.progress >= 35}
        />
        <ProgressStep
          text="Analyzing temporal context..."
          isActive={progressData && progressData.progress >= 35 && progressData.progress < 45}
          isComplete={progressData && progressData.progress >= 45}
        />
        <ProgressStep
          text="Fact-checking with external sources..."
          isActive={progressData && progressData.progress >= 45 && progressData.progress < 70}
          isComplete={progressData && progressData.progress >= 70}
        />
        <ProgressStep
          text="Generating final verdict..."
          isActive={progressData && progressData.progress >= 70 && progressData.progress < 85}
          isComplete={progressData && progressData.progress >= 85}
        />
        <ProgressStep
          text="Analyzing potential motives..."
          isActive={progressData && progressData.progress >= 85 && progressData.progress < 100}
          isComplete={progressData && progressData.progress >= 100}
        />
      </div>
    </div>
  )
}

function ProgressStep({ text, isActive, isComplete }) {
  return (
    <div className="flex items-center">
      <div className={`w-2 h-2 rounded-full mr-3 ${
        isComplete ? 'bg-green-500' :
        isActive ? 'bg-blue-500' :
        'bg-gray-300'
      }`}></div>
      <span className={
        isComplete ? 'text-green-600' :
        isActive ? 'text-blue-600' :
        'text-gray-600'
      }>
        {text}
      </span>
      {isComplete && (
        <svg className="w-4 h-4 ml-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      )}
    </div>
  )
}

export default LoadingState 
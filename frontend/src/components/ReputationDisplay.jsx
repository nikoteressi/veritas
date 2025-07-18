import React from 'react'

function ReputationDisplay({ reputation, warnings = [] }) {
  if (!reputation) {
    return null
  }

  const getAccuracyRate = () => {
    if (reputation.total_posts_checked === 0) return 0
    return Math.round(
      ((reputation.true_count + reputation.partially_true_count) / reputation.total_posts_checked) * 100
    )
  }

  const getFalseRate = () => {
    if (reputation.total_posts_checked === 0) return 0
    return Math.round((reputation.false_count / reputation.total_posts_checked) * 100)
  }

  const getReputationColor = () => {
    const falseRate = getFalseRate()
    if (falseRate >= 50) return 'text-red-600 bg-red-50 border-red-200'
    if (falseRate >= 30) return 'text-orange-600 bg-orange-50 border-orange-200'
    if (falseRate >= 15) return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    return 'text-green-600 bg-green-50 border-green-200'
  }

  const getReputationLabel = () => {
    const falseRate = getFalseRate()
    if (falseRate >= 50) return 'High Risk'
    if (falseRate >= 30) return 'Moderate Risk'
    if (falseRate >= 15) return 'Low Risk'
    return 'Reliable'
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        User Reputation: @{reputation.nickname}
      </h3>

      {/* Reputation Status */}
      <div className={`rounded-lg border p-4 mb-4 ${getReputationColor()}`}>
        <div className="flex items-center justify-between">
          <span className="font-medium">{getReputationLabel()}</span>
          <span className="text-sm">
            {getAccuracyRate()}% Accuracy Rate
          </span>
        </div>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="mb-4">
          {warnings.map((warning, index) => (
            <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-3 mb-2">
              <div className="flex items-start">
                <svg className="w-5 h-5 text-red-500 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <p className="text-red-700 text-sm">{warning}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">
            {reputation.total_posts_checked}
          </div>
          <div className="text-sm text-gray-600">Total Posts</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {getAccuracyRate()}%
          </div>
          <div className="text-sm text-gray-600">Accuracy</div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">True Posts</span>
          <div className="flex items-center">
            <span className="text-sm font-medium text-green-600 mr-2">
              {reputation.true_count}
            </span>
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full"
                style={{ 
                  width: `${reputation.total_posts_checked > 0 ? 
                    (reputation.true_count / reputation.total_posts_checked) * 100 : 0}%` 
                }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Partially True</span>
          <div className="flex items-center">
            <span className="text-sm font-medium text-yellow-600 mr-2">
              {reputation.partially_true_count}
            </span>
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-yellow-500 h-2 rounded-full"
                style={{ 
                  width: `${reputation.total_posts_checked > 0 ? 
                    (reputation.partially_true_count / reputation.total_posts_checked) * 100 : 0}%` 
                }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">False Posts</span>
          <div className="flex items-center">
            <span className="text-sm font-medium text-red-600 mr-2">
              {reputation.false_count}
            </span>
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-red-500 h-2 rounded-full"
                style={{ 
                  width: `${reputation.total_posts_checked > 0 ? 
                    (reputation.false_count / reputation.total_posts_checked) * 100 : 0}%` 
                }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Ironic/Satirical</span>
          <div className="flex items-center">
            <span className="text-sm font-medium text-blue-600 mr-2">
              {reputation.ironic_count}
            </span>
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full"
                style={{ 
                  width: `${reputation.total_posts_checked > 0 ? 
                    (reputation.ironic_count / reputation.total_posts_checked) * 100 : 0}%` 
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Timestamps */}
      {reputation.last_checked_date && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Last checked: {new Date(reputation.last_checked_date).toLocaleDateString()}
          </p>
        </div>
      )}
    </div>
  )
}

export default ReputationDisplay

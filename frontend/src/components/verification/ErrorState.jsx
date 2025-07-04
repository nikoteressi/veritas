import React from 'react'

function ErrorState({ error }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Verification Results
      </h2>

      {/* Error Banner */}
      <div className="rounded-lg border p-4 mb-6 text-red-600 bg-red-50 border-red-200">
        <div className="flex items-center">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          <span className="ml-2 font-medium">
            Verification Failed
          </span>
        </div>
        {error && (
          <p className="mt-2 text-sm">{error}</p>
        )}
      </div>

      <div className="text-center py-8">
        <svg
          className="mx-auto h-16 w-16 text-red-400 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
          />
        </svg>
        <p className="text-gray-600">
          Please try again or contact support if the problem persists
        </p>
      </div>
    </div>
  )
}

export default ErrorState 
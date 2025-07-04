import React from 'react'
import LoadingState from './verification/LoadingState'
import InitialState from './verification/InitialState'
import ErrorState from './verification/ErrorState'
import ResultDisplay from './verification/ResultDisplay'

function VerificationResults({ result, isLoading, progressData, error }) {
  // Show loading state when verification is in progress
  if (isLoading) {
    return <LoadingState progressData={progressData} />
  }

  // Show error state if there's an error or if result status is error
  if (error || (result && result.status === 'error')) {
    return <ErrorState error={error || result?.message} />
  }

  // Show result display when we have a successful result
  if (result) {
    return <ResultDisplay result={result} />
  }

  // Show initial state when no verification has been started
  return <InitialState />
}

export default VerificationResults

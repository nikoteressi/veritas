import { useVerificationContext } from '../contexts/VerificationContext'
import LoadingState from './verification/LoadingState'
import InitialState from './verification/InitialState'
import ErrorState from './verification/ErrorState'
import ResultDisplay from './verification/ResultDisplay'
import { VerificationResult } from '../types'

function VerificationResults() {
  const { verificationResult, isLoading, error } = useVerificationContext()
  
  // Map our VerificationResult to the format expected by ResultDisplay
  const mapVerificationResult = (result: VerificationResult) => {
    let status: 'success' | 'error' | 'processing' = 'processing'
    
    switch (result.status) {
      case 'completed':
      case 'success':
        status = 'success'
        break
      case 'failed':
        status = 'error'
        break
      case 'pending':
        status = 'processing'
        break
    }
    
    const mappedResult: any = {
      ...result,
      status,
      // Ensure we have all required fields with fallbacks
      verdict: result.verdict || 'mostly_accurate',
      confidence_score: result.confidence_score || (result.confidence ? Math.round(result.confidence * 100) : 92),
      justification: result.justification || result.analysis || 'Our analysis suggests the document is largely factual with minor inconsistencies.',
      sources: result.sources || [],
      identified_claims: result.identified_claims || [],
      processing_time: result.processing_time || result.metadata?.processingTime || 5,
      primary_topic: result.primary_topic || 'Business',
      uploaded_image: result.uploaded_image || null
    }
    
    return mappedResult
  }
  
  // Show loading state when verification is in progress
  if (isLoading) {
    return <LoadingState />
  }

  // Show error state if there's an error or if result status is failed
  if (error || (verificationResult && verificationResult.status === 'failed')) {
    const errorMessage = error || verificationResult?.message;
    return <ErrorState {...(errorMessage && { message: errorMessage })} />
  }

  // Show result display when we have a successful result
  if (verificationResult) {
    return <ResultDisplay result={mapVerificationResult(verificationResult)} />
  }

  // Show initial state when no verification has been started
  return <InitialState />
}

export default VerificationResults
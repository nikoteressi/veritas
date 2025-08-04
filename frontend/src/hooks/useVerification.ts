/**
 * Custom hook for handling verification requests.
 */
import { useState, useCallback } from 'react'
import { apiService } from '../services/apiService'
import { validateFile, validatePrompt, handleAPIError } from '../utils/errorHandling'
import { VerificationResult } from '../types'

interface ValidationErrors {
  file?: string
  prompt?: string
  [key: string]: string | undefined
}

interface UseVerificationReturn {
  isLoading: boolean
  error: string | null
  validationErrors: ValidationErrors
  submitVerification: (
    file: File | null,
    prompt: string,
    sessionId: string | null,
    isWebSocketConnected: boolean,
    onStart?: () => void,
    onComplete?: (result: VerificationResult) => void
  ) => Promise<void>
  resetState: () => void
  setIsLoading: (loading: boolean) => void
}

export const useVerification = (): UseVerificationReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({})

  const submitVerification = useCallback(async (
    file: File | null,
    prompt: string,
    sessionId: string | null,
    isWebSocketConnected: boolean,
    onStart?: () => void,
    onComplete?: (result: VerificationResult) => void
  ): Promise<void> => {
    setError(null)
    setValidationErrors({})

    try {
      // Validate inputs
      if (!file) {
        setValidationErrors({ file: 'Please select an image file' })
        return
      }

      validateFile(file)
      const validatedPrompt = validatePrompt(prompt)

      // Start loading state
      setIsLoading(true)
      if (onStart) {
        onStart()
      }

      // Submit the verification request
      const response = await apiService.submitVerificationRequest({
        file,
        prompt: validatedPrompt,
        sessionId: isWebSocketConnected ? sessionId : null
      })

      // Handle response - if WebSocket is connected, the result will come via WebSocket
      // Otherwise, handle the response directly
      if (!isWebSocketConnected || !sessionId) {
        setIsLoading(false)
        if (onComplete) {
          onComplete(response.data)
        }
      } else {
        console.log('WebSocket connected - waiting for WebSocket messages')
      }
      // If WebSocket is connected, loading state will be handled by WebSocket messages

    } catch (error) {
      console.error('Verification error:', error)
      setIsLoading(false)

      try {
        handleAPIError(error as any)
      } catch (handledError) {
        const errorMessage = handledError instanceof Error 
          ? handledError.message 
          : 'An error occurred during verification'
        setError(errorMessage)

        if (onComplete) {
          onComplete({
            status: 'failed',
            message: errorMessage,
            timestamp: new Date().toISOString()
          })
        }
      }
    }
  }, [])

  const resetState = useCallback(() => {
    setError(null)
    setValidationErrors({})
    setIsLoading(false)
  }, [])

  return {
    isLoading,
    error,
    validationErrors,
    submitVerification,
    resetState,
    setIsLoading
  }
}
/**
 * Custom hook for handling verification requests.
 */
import { useState, useCallback } from 'react';
import { apiService } from '../services/apiService';
import { validateFile, validatePrompt, handleAPIError } from '../utils/errorHandling';

export const useVerification = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState({});

  const submitVerification = useCallback(async (
    file,
    prompt,
    sessionId,
    isWebSocketConnected,
    onStart,
    onComplete
  ) => {
    setError(null);
    setValidationErrors({});

    try {
      // Validate inputs
      if (!file) {
        setValidationErrors({ file: 'Please select an image file' });
        return;
      }

      validateFile(file);
      const validatedPrompt = validatePrompt(prompt);

      // Start loading state
      setIsLoading(true);
      if (onStart) {
        onStart();
      }

      // Submit the verification request
      const response = await apiService.submitVerificationRequest({
        file,
        prompt: validatedPrompt,
        sessionId: isWebSocketConnected ? sessionId : null
      });

      // Handle response - if WebSocket is connected, the result will come via WebSocket
      // Otherwise, handle the response directly
      if (!isWebSocketConnected || !sessionId) {
        setIsLoading(false);
        if (onComplete) {
          onComplete(response.data);
        }
      } else {
        console.log('WebSocket connected - waiting for WebSocket messages');
      }
      // If WebSocket is connected, loading state will be handled by WebSocket messages

    } catch (error) {
      console.error('Verification error:', error);
      setIsLoading(false);

      try {
        handleAPIError(error);
      } catch (handledError) {
        const errorMessage = handledError.message || 'An error occurred during verification';
        setError(errorMessage);

        if (onComplete) {
          onComplete({
            status: 'error',
            message: errorMessage
          });
        }
      }
    }
  }, []);

  const resetState = useCallback(() => {
    setError(null);
    setValidationErrors({});
    setIsLoading(false);
  }, []);

  return {
    isLoading,
    error,
    validationErrors,
    submitVerification,
    resetState,
    setIsLoading
  };
};
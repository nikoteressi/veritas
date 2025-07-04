/**
 * Hook for using verification state service with React components.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { verificationStateService } from '../services/verificationStateService';

export const useVerificationState = () => {
  const [state, setState] = useState(() => verificationStateService.getState());
  const unsubscribeRef = useRef(null);

  // Subscribe to state changes
  useEffect(() => {
    unsubscribeRef.current = verificationStateService.subscribe((newState) => {
      setState(newState);
    });

    // Cleanup subscription on unmount
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    };
  }, []);

  /**
   * Start a verification process.
   * @param {Object} options - Verification options
   */
  const startVerification = (options = {}) => {
    verificationStateService.startVerification(options);
  };

  /**
   * Update verification progress.
   * @param {Object} progressData - Progress information
   */
  const updateProgress = (progressData) => {
    verificationStateService.updateProgress(progressData);
  };

  /**
   * Complete verification with result.
   * @param {Object} result - Verification result
   */
  const completeVerification = (result) => {
    verificationStateService.completeVerification(result);
  };

  /**
   * Set verification error.
   * @param {string|Error} error - Error message or Error object
   */
  const setError = (error) => {
    verificationStateService.setError(error);
  };

  /**
   * Clear verification state.
   */
  const clearState = () => {
    verificationStateService.clearState();
  };

  /**
   * Reset verification state to initial state.
   */
  const reset = () => {
    verificationStateService.reset();
  };

  /**
   * Handle WebSocket message for verification updates.
   * @param {string} messageType - Type of WebSocket message
   * @param {*} data - Message data
   */
  const handleWebSocketMessage = useCallback((messageType, data) => {
    verificationStateService.handleWebSocketMessage(messageType, data);
  }, []);

  /**
   * Get verification summary for display.
   * @returns {Object} Verification summary
   */
  const getVerificationSummary = () => {
    return verificationStateService.getVerificationSummary();
  };

  return {
    // State
    verificationResult: state.verificationResult,
    isLoading: state.isLoading,
    progressData: state.progressData,
    error: state.error,
    
    // Actions
    startVerification,
    updateProgress,
    completeVerification,
    setError,
    clearState,
    reset,
    handleWebSocketMessage,
    
    // Utilities
    getVerificationSummary,
    
    // Service instance (for advanced usage)
    verificationStateService
  };
}; 
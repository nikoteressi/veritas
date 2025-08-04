/**
 * Hook for using verification state service with React components.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { VerificationResult, VerificationState } from '../types';
import { verificationStateService } from '../services/verificationStateService';

interface VerificationSummary {
  hasResult: boolean;
  isProcessing: boolean;
  hasError: boolean;
  status: 'idle' | 'loading' | 'completed' | 'error';
}

interface UseVerificationStateReturn extends VerificationState {
  startVerification: (options?: Record<string, any>) => void;
  completeVerification: (result: VerificationResult) => void;
  setError: (error: string | Error | null) => void;
  clearState: () => void;
  reset: () => void;
  handleWebSocketMessage: (messageType: string, data: any) => void;
  getVerificationSummary: () => VerificationSummary;
  verificationStateService: typeof verificationStateService;
}

export const useVerificationState = (): UseVerificationStateReturn => {
  const [state, setState] = useState<VerificationState>(() => verificationStateService.getState());
  const unsubscribeRef = useRef<(() => void) | null>(null);

  // Subscribe to state changes
  useEffect(() => {
    unsubscribeRef.current = verificationStateService.subscribe((newState: VerificationState) => {
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
   * @param options - Verification options
   */
  const startVerification = (options: Record<string, any> = {}): void => {
    verificationStateService.startVerification(options);
  };

  /**
   * Complete verification with result.
   * @param result - Verification result
   */
  const completeVerification = (result: VerificationResult): void => {
    verificationStateService.completeVerification(result);
  };

  /**
   * Set verification error.
   * @param error - Error message or Error object
   */
  const setError = (error: string | Error | null): void => {
    if (error !== null) {
      verificationStateService.setError(error);
    }
  };

  /**
   * Clear verification state.
   */
  const clearState = (): void => {
    verificationStateService.clearState();
  };

  /**
   * Reset verification state to initial state.
   */
  const reset = (): void => {
    verificationStateService.reset();
  };

  /**
   * Handle WebSocket message for verification updates.
   * @param messageType - Type of WebSocket message
   * @param data - Message data
   */
  const handleWebSocketMessage = useCallback((messageType: string, data: any): void => {
    verificationStateService.handleWebSocketMessage(messageType, data);
  }, []);

  /**
   * Get verification summary for display.
   * @returns Verification summary
   */
  const getVerificationSummary = (): VerificationSummary => {
    return verificationStateService.getVerificationSummary();
  };

  return {
    // State
    verificationResult: state.verificationResult,
    isLoading: state.isLoading,
    error: state.error,
    ...(state.progress !== undefined && { progress: state.progress }),
    
    // Actions
    startVerification,
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
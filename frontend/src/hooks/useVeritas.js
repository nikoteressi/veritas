/**
 * Consolidated hook for managing Veritas application state.
 * This hook combines WebSocket connectivity and verification state management
 * into a single, clean interface for the main App component.
 */
import { useEffect, useCallback } from 'react';
import { useWebSocketService } from './useWebSocketService';
import { useVerificationState } from './useVerificationState';
import logger from '../utils/logger';

export const useVeritas = () => {
  // WebSocket connection management
  const {
    isConnected,
    sessionId,
    lastMessage,
    sendMessage,
    subscribeToMessage,
    reconnect,
    getConnectionInfo,
    webSocketService
  } = useWebSocketService();

  // Verification state management
  const {
    verificationResult,
    isLoading,
    progressData,
    error,
    startVerification,
    completeVerification,
    setError,
    clearState,
    reset,
    handleWebSocketMessage,
    getVerificationSummary,
    verificationStateService
  } = useVerificationState();

  // Handle WebSocket messages and delegate to appropriate handlers
  useEffect(() => {
    if (lastMessage) {
      // Handle verification-related messages
      const verificationMessageTypes = ['progress_update', 'verification_result', 'error', 'verification_started'];
      
      if (verificationMessageTypes.includes(lastMessage.type)) {
        handleWebSocketMessage(lastMessage.type, lastMessage.data);
      } else {
        // Handle other message types
        switch (lastMessage.type) {
          case 'connection_established':
            logger.info('WebSocket connection established:', lastMessage.data);
            break;
          case 'session_established':
            logger.info('WebSocket session established:', lastMessage.data);
            break;
          case 'pong':
            // Skip logging pong messages to reduce console noise
            break;
          case 'status_response':
            logger.debug('WebSocket status response:', lastMessage.data);
            break;
          case 'echo':
            logger.debug('WebSocket echo received:', lastMessage.data);
            break;
          default:
            logger.warn('Unhandled WebSocket message:', lastMessage);
        }
      }
    }
  }, [lastMessage, handleWebSocketMessage]);

  // Callback for handling verification completion
  const handleVerificationComplete = useCallback((result) => {
    completeVerification(result);
  }, [completeVerification]);

  // Callback for handling verification start
  const handleVerificationStart = useCallback(() => {
    startVerification();
  }, [startVerification]);

  // Submit verification function that can be used by components
  const submitVerification = useCallback(async (file, prompt, sessionId, isWebSocketConnected) => {
    try {
      // Start verification state
      handleVerificationStart();
      
      // Here you would typically call your verification service
      // For now, we'll just log the action
      logger.info('Submitting verification:', { file: file?.name, prompt, sessionId, isWebSocketConnected });
      
      // You might want to send a WebSocket message to start verification
      if (isWebSocketConnected && sessionId) {
        sendMessage('start_verification', {
          filename: file?.name,
          prompt,
          sessionId
        });
      }
      
    } catch (error) {
      logger.error('Error submitting verification:', error);
      setError(error.message || 'Failed to submit verification');
    }
  }, [handleVerificationStart, sendMessage, setError]);

  // Clean interface for the App component
  return {
    // Connection status
    isConnected,
    sessionId,
    
    // Verification state
    verificationResult,
    isLoading,
    progressData,
    error,
    
    // Actions
    submitVerification,
    onVerificationStart: handleVerificationStart,
    onVerificationComplete: handleVerificationComplete,
    clearState,
    reset,
    
    // WebSocket actions
    sendMessage,
    subscribeToMessage,
    reconnect,
    getConnectionInfo,
    
    // Utilities
    getVerificationSummary,
    
    // Service instances for advanced usage
    webSocketService,
    verificationStateService
  };
}; 
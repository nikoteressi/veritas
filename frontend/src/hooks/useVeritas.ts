/**
 * Simplified hook for managing core Veritas application state.
 * Focuses on essential functionality with clear separation of concerns.
 */
import { useCallback, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useVerificationState } from './useVerificationState';
import { UseVeritasReturn, VerificationResult } from '../types';
import logger from '../utils/logger';

export const useVeritas = (): UseVeritasReturn => {
  // WebSocket connection management
  const {
    isConnected,
    sessionId,
    reconnectAttempts,
    sendMessage,
    subscribeToMessage,
    reconnect,
    getConnectionInfo
  } = useWebSocketContext();

  // Verification state management
  const {
    verificationResult,
    isLoading,
    error,
    startVerification,
    completeVerification,
    setError,
    clearState,
    reset,
    getVerificationSummary
  } = useVerificationState();

  // Мемоизированный статус подключения
  const connectionStatus = useMemo(() => 
    isConnected ? 'connected' : 'disconnected', 
    [isConnected]
  );

  // Упрощённая функция отправки верификации
  const submitVerification = useCallback(async (
    file: File, 
    prompt: string
  ): Promise<void> => {
    try {
      startVerification();
      
      logger.info('Submitting verification:', { 
        file: file?.name, 
        prompt, 
        sessionId 
      });
      
      if (isConnected && sessionId) {
        sendMessage('start_verification', {
          filename: file?.name,
          prompt,
          sessionId
        });
      } else {
        throw new Error('WebSocket not connected');
      }
      
    } catch (error) {
      logger.error('Error submitting verification:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to submit verification';
      setError(errorMessage);
    }
  }, [startVerification, sendMessage, setError, isConnected, sessionId]);

  // Упрощённые обработчики
  const handleVerificationComplete = useCallback((result: VerificationResult) => {
    completeVerification(result);
  }, [completeVerification]);

  // Чистый интерфейс для компонентов
  return {
    // Статус подключения
    isConnected,
    sessionId,
    connectionStatus,
    reconnectAttempts, // Add missing property
    
    // Состояние верификации
    verificationResult,
    isLoading,
    error,
    
    // Основные действия
    submitVerification,
    onVerificationStart: startVerification, // Add missing property
    onVerificationComplete: handleVerificationComplete,
    clearState,
    reset,
    
    // WebSocket действия
    sendMessage,
    subscribeToMessage, // Add missing property
    reconnect,
    getConnectionInfo, // Add missing property
    
    // Утилиты
    getVerificationSummary
  };
};
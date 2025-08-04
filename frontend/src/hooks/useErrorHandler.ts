import { useCallback, useState } from 'react';
import { AppError, ErrorSeverity } from '../types';
import { errorService } from '../services/errorService';

export interface UseErrorHandlerReturn {
  error: AppError | null;
  hasError: boolean;
  clearError: () => void;
  handleError: (error: Error | AppError, context?: Record<string, unknown>) => void;
  handleAsyncError: <T>(
    operation: () => Promise<T>,
    context?: string
  ) => Promise<T | null>;
  handleSyncError: <T>(
    operation: () => T,
    context?: string
  ) => T | null;
  createError: (
    message: string,
    code?: string,
    severity?: ErrorSeverity,
    details?: Record<string, unknown>
  ) => AppError;
}

/**
 * Custom hook for handling errors in React components
 * Provides centralized error handling with logging and state management
 */
export function useErrorHandler(): UseErrorHandlerReturn {
  const [error, setError] = useState<AppError | null>(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const handleError = useCallback((
    error: Error | AppError,
    context?: Record<string, unknown>
  ) => {
    const appError = error instanceof Error 
      ? errorService.fromError(error) 
      : error;
    
    setError(appError);
    errorService.logError(appError, context);
  }, []);

  const handleAsyncError = useCallback(async <T>(
    operation: () => Promise<T>,
    context?: string
  ): Promise<T | null> => {
    try {
      clearError();
      return await operation();
    } catch (error) {
      const appError = errorService.fromError(
        error instanceof Error ? error : new Error(String(error)),
        'high'
      );
      
      setError(appError);
      errorService.logError(appError, { context, operation: 'async' });
      return null;
    }
  }, [clearError]);

  const handleSyncError = useCallback(<T>(
    operation: () => T,
    context?: string
  ): T | null => {
    try {
      clearError();
      return operation();
    } catch (error) {
      const appError = errorService.fromError(
        error instanceof Error ? error : new Error(String(error)),
        'medium'
      );
      
      setError(appError);
      errorService.logError(appError, { context, operation: 'sync' });
      return null;
    }
  }, [clearError]);

  const createError = useCallback((
    message: string,
    code?: string,
    severity: ErrorSeverity = 'medium',
    details?: Record<string, unknown>
  ): AppError => {
    return errorService.createError(message, code, severity, details);
  }, []);

  return {
    error,
    hasError: error !== null,
    clearError,
    handleError,
    handleAsyncError,
    handleSyncError,
    createError
  };
}

/**
 * Hook for handling specific error types with predefined configurations
 */
export function useSpecificErrorHandler(defaultSeverity: ErrorSeverity = 'medium') {
  const { handleError, ...rest } = useErrorHandler();

  const handleNetworkError = useCallback((
    message: string,
    status?: number,
    context?: Record<string, unknown>
  ) => {
    const error = errorService.createError(
      message,
      'NETWORK_ERROR',
      'high',
      { status, type: 'network' }
    );
    handleError(error, context);
  }, [handleError]);

  const handleValidationError = useCallback((
    message: string,
    field?: string,
    context?: Record<string, unknown>
  ) => {
    const error = errorService.createError(
      message,
      'VALIDATION_ERROR',
      defaultSeverity,
      { field, type: 'validation' }
    );
    handleError(error, context);
  }, [handleError, defaultSeverity]);

  const handleFileError = useCallback((
    message: string,
    fileName?: string,
    context?: Record<string, unknown>
  ) => {
    const error = errorService.createError(
      message,
      'FILE_ERROR',
      defaultSeverity,
      { fileName, type: 'file' }
    );
    handleError(error, context);
  }, [handleError, defaultSeverity]);

  const handleWebSocketError = useCallback((
    message: string,
    event?: string,
    context?: Record<string, unknown>
  ) => {
    const error = errorService.createError(
      message,
      'WEBSOCKET_ERROR',
      'high',
      { event, type: 'websocket' }
    );
    handleError(error, context);
  }, [handleError]);

  return {
    ...rest,
    handleError,
    handleNetworkError,
    handleValidationError,
    handleFileError,
    handleWebSocketError
  };
}
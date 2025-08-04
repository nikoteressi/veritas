import { AppError, ErrorSeverity } from '../types';

/**
 * Centralized error service for handling, logging, and reporting errors
 */
export class ErrorService {
  private static instance: ErrorService;
  private errorQueue: AppError[] = [];
  private maxQueueSize = 100;

  private constructor() {}

  public static getInstance(): ErrorService {
    if (!ErrorService.instance) {
      ErrorService.instance = new ErrorService();
    }
    return ErrorService.instance;
  }

  /**
   * Create a standardized AppError
   */
  public createError(
    message: string,
    code?: string,
    severity: ErrorSeverity = 'medium',
    details?: Record<string, unknown>,
    cause?: Error
  ): AppError {
    const appError: AppError = {
      message,
      code: code || 'UNKNOWN_ERROR',
      severity,
      timestamp: new Date().toISOString(),
      details: details || {}
    };
    
    if (cause) {
      appError.cause = cause;
    }
    
    return appError;
  }

  /**
   * Log error to console and external services
   */
  public logError(error: AppError | Error, context?: Record<string, unknown>): void {
    const appError = error instanceof Error ? this.fromError(error) : error;
    
    // Add to queue
    this.addToQueue(appError);

    // Console logging with appropriate level
    const logLevel = this.getLogLevel(appError.severity);
    const logData = {
      ...appError,
      context,
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: appError.timestamp
    };

    console[logLevel]('Error logged:', logData);

    // Send to external service in production
    if (import.meta.env.MODE === 'production') {
      this.reportToExternalService(appError, context);
    }
  }

  /**
   * Convert native Error to AppError
   */
  public fromError(error: Error, severity: ErrorSeverity = 'high'): AppError {
    return this.createError(
      error.message,
      error.name,
      severity,
      {
        stack: error.stack,
        name: error.name
      },
      error
    );
  }

  /**
   * Handle async errors with proper logging
   */
  public async handleAsyncError<T>(
    operation: () => Promise<T>,
    context?: string
  ): Promise<T | null> {
    try {
      return await operation();
    } catch (error) {
      const appError = this.fromError(
        error instanceof Error ? error : new Error(String(error)),
        'high'
      );
      
      this.logError(appError, { context, operation: 'async' });
      return null;
    }
  }

  /**
   * Handle sync errors with proper logging
   */
  public handleSyncError<T>(
    operation: () => T,
    context?: string
  ): T | null {
    try {
      return operation();
    } catch (error) {
      const appError = this.fromError(
        error instanceof Error ? error : new Error(String(error)),
        'medium'
      );
      
      this.logError(appError, { context, operation: 'sync' });
      return null;
    }
  }

  /**
   * Get recent errors for debugging
   */
  public getRecentErrors(count = 10): AppError[] {
    return this.errorQueue.slice(-count);
  }

  /**
   * Clear error queue
   */
  public clearErrors(): void {
    this.errorQueue = [];
  }

  /**
   * Get error statistics
   */
  public getErrorStats(): {
    total: number;
    bySeverity: Record<ErrorSeverity, number>;
    recent: number;
  } {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    
    const bySeverity = this.errorQueue.reduce((acc, error) => {
      acc[error.severity] = (acc[error.severity] || 0) + 1;
      return acc;
    }, {} as Record<ErrorSeverity, number>);

    const recent = this.errorQueue.filter(
      error => new Date(error.timestamp) > oneHourAgo
    ).length;

    return {
      total: this.errorQueue.length,
      bySeverity,
      recent
    };
  }

  private addToQueue(error: AppError): void {
    this.errorQueue.push(error);
    
    // Maintain queue size
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue = this.errorQueue.slice(-this.maxQueueSize);
    }
  }

  private getLogLevel(severity: ErrorSeverity): 'error' | 'warn' | 'info' {
    switch (severity) {
      case 'critical':
      case 'high':
        return 'error';
      case 'medium':
        return 'warn';
      case 'low':
        return 'info';
      default:
        return 'error';
    }
  }

  private async reportToExternalService(
    error: AppError,
    context?: Record<string, unknown>
  ): Promise<void> {
    try {
      // TODO: Implement external error reporting service integration
      // Examples: Sentry, LogRocket, Bugsnag, etc.
      
      const payload = {
        error,
        context,
        environment: import.meta.env.MODE,
        version: import.meta.env.VITE_APP_VERSION || 'unknown',
        userId: this.getUserId(),
        sessionId: this.getSessionId()
      };

      // For now, just log to console in production
      console.error('Would report to external service:', payload);
      
      // Example implementation:
      // await fetch('/api/errors', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(payload)
      // });
    } catch (reportError) {
      console.error('Failed to report error to external service:', reportError);
    }
  }

  private getUserId(): string | null {
    // TODO: Get user ID from authentication context
    return null;
  }

  private getSessionId(): string | null {
    // TODO: Get session ID from WebSocket context or local storage
    return sessionStorage.getItem('sessionId');
  }
}

// Export singleton instance
export const errorService = ErrorService.getInstance();

// Utility functions for common error scenarios
export const createNetworkError = (message: string, status?: number): AppError => {
  return errorService.createError(
    message,
    'NETWORK_ERROR',
    'high',
    { status, type: 'network' }
  );
};

export const createValidationError = (message: string, field?: string): AppError => {
  return errorService.createError(
    message,
    'VALIDATION_ERROR',
    'medium',
    { field, type: 'validation' }
  );
};

export const createFileError = (message: string, fileName?: string): AppError => {
  return errorService.createError(
    message,
    'FILE_ERROR',
    'medium',
    { fileName, type: 'file' }
  );
};

export const createWebSocketError = (message: string, event?: string): AppError => {
  return errorService.createError(
    message,
    'WEBSOCKET_ERROR',
    'high',
    { event, type: 'websocket' }
  );
};
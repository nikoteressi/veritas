import { memo } from 'react';
import { ExclamationTriangleIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { AppError, ErrorSeverity } from '../../types';

export interface ErrorDisplayProps {
  error: AppError | string | null;
  onDismiss?: () => void;
  showDetails?: boolean;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * ErrorDisplay component for consistent error presentation
 * Supports both AppError objects and simple string messages
 */
export const ErrorDisplay = memo<ErrorDisplayProps>(({
  error,
  onDismiss,
  showDetails = false,
  className = '',
  size = 'md'
}) => {
  if (!error) return null;

  const appError = typeof error === 'string' 
    ? { message: error, severity: 'medium' as ErrorSeverity, code: 'UNKNOWN', timestamp: new Date().toISOString(), details: {} }
    : error;

  const getSeverityStyles = (severity: ErrorSeverity): string => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 border-red-300 text-red-800';
      case 'high':
        return 'bg-red-50 border-red-200 text-red-700';
      case 'medium':
        return 'bg-orange-50 border-orange-200 text-orange-700';
      case 'low':
        return 'bg-yellow-50 border-yellow-200 text-yellow-700';
      default:
        return 'bg-red-50 border-red-200 text-red-700';
    }
  };

  const getIconColor = (severity: ErrorSeverity): string => {
    switch (severity) {
      case 'critical':
        return 'text-red-600';
      case 'high':
        return 'text-red-500';
      case 'medium':
        return 'text-orange-500';
      case 'low':
        return 'text-yellow-500';
      default:
        return 'text-red-500';
    }
  };

  const getSizeStyles = (size: 'sm' | 'md' | 'lg'): string => {
    switch (size) {
      case 'sm':
        return 'p-3 text-sm';
      case 'md':
        return 'p-4 text-sm';
      case 'lg':
        return 'p-6 text-base';
      default:
        return 'p-4 text-sm';
    }
  };

  const severityStyles = getSeverityStyles(appError.severity);
  const iconColor = getIconColor(appError.severity);
  const sizeStyles = getSizeStyles(size);

  return (
    <div className={`border rounded-lg ${severityStyles} ${sizeStyles} ${className}`}>
      <div className="flex items-start">
        <ExclamationTriangleIcon 
          className={`w-5 h-5 ${iconColor} mt-0.5 mr-3 flex-shrink-0`} 
        />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="font-medium">{appError.message}</p>
              
              {appError.code && appError.code !== 'UNKNOWN' && (
                <p className="mt-1 text-xs opacity-75">
                  Error Code: {appError.code}
                </p>
              )}
              
              {showDetails && appError.details && Object.keys(appError.details).length > 0 && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs font-medium opacity-75 hover:opacity-100">
                    Technical Details
                  </summary>
                  <div className="mt-2 p-2 bg-black bg-opacity-5 rounded text-xs font-mono">
                    <pre className="whitespace-pre-wrap">
                      {JSON.stringify(appError.details, null, 2)}
                    </pre>
                  </div>
                </details>
              )}
              
              {showDetails && appError.timestamp && (
                <p className="mt-1 text-xs opacity-75">
                  {new Date(appError.timestamp).toLocaleString()}
                </p>
              )}
            </div>
            
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="ml-3 flex-shrink-0 p-1 rounded-md hover:bg-black hover:bg-opacity-10 transition-colors"
                aria-label="Dismiss error"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

ErrorDisplay.displayName = 'ErrorDisplay';

export default ErrorDisplay;
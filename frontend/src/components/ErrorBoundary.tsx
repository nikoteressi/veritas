import { Component, ReactNode, ErrorInfo } from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showDetails?: boolean;
  level?: 'page' | 'component' | 'section';
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
}

/**
 * Error Boundary Component
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI instead of the component tree that crashed.
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return { 
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Report to error tracking service
    this.reportError(error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: ErrorInfo): void => {
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      level: this.props.level || 'component',
      errorId: this.state.errorId,
    };

    console.error('Error Report:', errorReport);
    // In production, send to error tracking service
  };

  private handleTryAgain = (): void => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      errorId: null,
    });
  };

  private handleRefresh = (): void => {
    window.location.reload();
  };

  private renderErrorDetails = (): ReactNode => {
    const { error, errorInfo, errorId } = this.state;
    const { showDetails = false } = this.props;

    if (!showDetails || !error) return null;

    return (
      <details className="mt-4 p-4 bg-gray-100 rounded-lg">
        <summary className="cursor-pointer font-medium text-gray-700 mb-2">
          Technical Details
        </summary>
        <div className="space-y-3 text-sm">
          <div>
            <strong>Error ID:</strong> {errorId}
          </div>
          <div>
            <strong>Error Message:</strong>
            <pre className="mt-1 p-2 bg-gray-200 rounded text-xs overflow-auto">
              {error.message}
            </pre>
          </div>
          {error.stack && (
            <div>
              <strong>Stack Trace:</strong>
              <pre className="mt-1 p-2 bg-gray-200 rounded text-xs overflow-auto max-h-32">
                {error.stack}
              </pre>
            </div>
          )}
          {errorInfo?.componentStack && (
            <div>
              <strong>Component Stack:</strong>
              <pre className="mt-1 p-2 bg-gray-200 rounded text-xs overflow-auto max-h-32">
                {errorInfo.componentStack}
              </pre>
            </div>
          )}
        </div>
      </details>
    );
  };

  private renderFallbackUI = (): ReactNode => {
    const { level = 'component' } = this.props;
    const isPageLevel = level === 'page';
    const containerClass = isPageLevel 
      ? 'min-h-screen flex items-center justify-center bg-gray-50'
      : 'p-6 bg-red-50 border border-red-200 rounded-lg';

    return (
      <div className={containerClass}>
        <div className="text-center max-w-md mx-auto">
          <div className="mb-4">
            <ExclamationTriangleIcon 
              className={`mx-auto ${isPageLevel ? 'h-16 w-16' : 'h-12 w-12'} text-red-500`}
              aria-hidden="true"
            />
          </div>
          
          <h2 className={`${isPageLevel ? 'text-2xl' : 'text-lg'} font-bold text-gray-900 mb-2`}>
            {isPageLevel ? 'Something went wrong' : 'Component Error'}
          </h2>
          
          <p className="text-gray-600 mb-6">
            {isPageLevel 
              ? 'We encountered an unexpected error. Please try refreshing the page or contact support if the problem persists.'
              : 'This component encountered an error and could not be displayed properly.'
            }
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={this.handleTryAgain}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <ArrowPathIcon className="w-4 h-4 mr-2" />
              Try Again
            </button>
            
            {isPageLevel && (
              <button
                onClick={this.handleRefresh}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Reload Page
              </button>
            )}
          </div>

          {(import.meta.env.DEV || this.props.showDetails) && this.renderErrorDetails()}
        </div>
      </div>
    );
  };

  override render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return this.renderFallbackUI();
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
import React, { Suspense } from 'react';
import { useTranslation } from 'react-i18next';
import WebSocketStatus from './components/WebSocketStatus';
import ErrorBoundary from './components/ErrorBoundary';
import PageLayout from './components/layout/PageLayout';
import { VerificationProvider, useVerificationContext } from './contexts/VerificationContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { ProgressProvider } from './contexts/ProgressContext';
import { useProgressInterpreter } from './hooks/useProgressInterpreter';

// Lazy load heavy components
const UploadForm = React.lazy(() => import('./components/UploadForm'));
const VerificationResults = React.lazy(() => import('./components/VerificationResults'));

interface PageContent {
  title: string;
  subtitle: string | null;
  showUpload: boolean;
}

function AppContent(): React.ReactElement {
  const { t } = useTranslation();
  const {
    verificationResult,
    isLoading
  } = useVerificationContext();
  
  // Initialize progress interpreter to handle WebSocket messages
  useProgressInterpreter();

  // Determine page title and layout based on current state
  const getPageContent = (): PageContent => {
    if (isLoading) {
      return {
        title: t('app.title.loading'),
        subtitle: t('app.subtitle.loading'),
        showUpload: false
      };
    }
    
    if (verificationResult) {
      return {
        title: t('app.title.complete'),
        subtitle: t('app.subtitle.complete'),
        showUpload: false
      };
    }
    
    return {
      title: t('app.title.default'),
      subtitle: t('app.subtitle.default'),
      showUpload: true
    };
  };

  const { title, subtitle, showUpload } = getPageContent();

  const LoadingSpinner = (): React.ReactElement => (
    <div className="flex items-center justify-center py-12">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span className="ml-3 text-gray-600">{t('common.loading')}</span>
    </div>
  );

  return (
    <PageLayout title={title} {...(subtitle && { subtitle })}>
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded-md z-50"
      >
        {t('accessibility.skipToContent')}
      </a>

      {/* WebSocket Status - positioned absolutely in top right */}
      <div className="fixed top-20 right-4 z-10">
        <WebSocketStatus />
      </div>

      {/* Main Content */}
      <main id="main-content" className="max-w-4xl mx-auto space-y-8">
        {showUpload && (
          <Suspense fallback={<LoadingSpinner />}>
            <UploadForm />
          </Suspense>
        )}
        
        {!showUpload && (
          <Suspense fallback={<LoadingSpinner />}>
            <VerificationResults />
          </Suspense>
        )}
      </main>
    </PageLayout>
  );
}

function App(): React.ReactElement {
  const handleError = (error: Error, errorInfo: React.ErrorInfo): void => {
    // Log to external service in production
    if (import.meta.env.MODE === 'production') {
      // TODO: Send to error tracking service (e.g., Sentry)
      console.error('Application Error:', {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      });
    }
  };

  return (
    <ErrorBoundary
      level="page"
      onError={handleError}
      showDetails={import.meta.env.DEV}
    >
      <WebSocketProvider>
        <ProgressProvider>
          <VerificationProvider>
            <ErrorBoundary level="section" showDetails={import.meta.env.DEV}>
              <AppContent />
            </ErrorBoundary>
          </VerificationProvider>
        </ProgressProvider>
      </WebSocketProvider>
    </ErrorBoundary>
  );
}

export default App;
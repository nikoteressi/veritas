import React from 'react'
import UploadForm from './components/UploadForm'
import VerificationResults from './components/VerificationResults'
import WebSocketStatus from './components/WebSocketStatus'
import ErrorBoundary from './components/ErrorBoundary'
import PageLayout from './components/layout/PageLayout'
import { useVeritas } from './hooks/useVeritas'

function App() {
  // Consolidated state management using the new useVeritas hook
  const {
    isConnected,
    sessionId,
    verificationResult,
    isLoading,
    error,
    onVerificationStart,
    onVerificationComplete
  } = useVeritas();

  // Determine page title and layout based on current state
  const getPageContent = () => {
    if (isLoading) {
      return {
        title: "Fact-Checking in Progress",
        subtitle: "We're analyzing your request. Please wait a moment.",
        showUpload: false
      }
    }
    
    if (verificationResult) {
      return {
        title: "Verification Complete",
        subtitle: null,
        showUpload: false
      }
    }
    
    return {
      title: "Verify a Post",
      subtitle: "Upload a screenshot and ask a question to check the facts.",
      showUpload: true
    }
  }

  const { title, subtitle, showUpload } = getPageContent()

  return (
    <ErrorBoundary>
      <PageLayout title={title} subtitle={subtitle}>
        {/* WebSocket Status - positioned absolutely in top right */}
        <div className="fixed top-20 right-4 z-10">
          <WebSocketStatus isConnected={isConnected} sessionId={sessionId} />
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-8">
          {showUpload && (
            <UploadForm
              onVerificationStart={onVerificationStart}
              onVerificationComplete={onVerificationComplete}
              isLoading={isLoading}
              sessionId={sessionId}
              isWebSocketConnected={isConnected}
            />
          )}
          
          {!showUpload && (
            <VerificationResults
              result={verificationResult}
              isLoading={isLoading}
              error={error}
            />
          )}
        </div>
      </PageLayout>
    </ErrorBoundary>
  )
}

export default App

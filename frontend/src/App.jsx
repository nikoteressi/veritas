import React from 'react'
import UploadForm from './components/UploadForm'
import VerificationResults from './components/VerificationResults'
import WebSocketStatus from './components/WebSocketStatus'
import ErrorBoundary from './components/ErrorBoundary'
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

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
        <header className="text-center mb-8">
          <div className="flex justify-between items-center mb-4">
            <div></div>
            <h1 className="text-4xl font-bold text-gray-900">
              Veritas
            </h1>
            <WebSocketStatus isConnected={isConnected} sessionId={sessionId} />
          </div>
          <p className="text-lg text-gray-600">
            AI-Powered Social Post Verifier
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <UploadForm
              onVerificationStart={onVerificationStart}
              onVerificationComplete={onVerificationComplete}
              isLoading={isLoading}
              sessionId={sessionId}
              isWebSocketConnected={isConnected}
            />
          </div>

          <div className="space-y-6">
            <VerificationResults
              result={verificationResult}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>

        <footer className="text-center mt-12 text-gray-500">
          <p>Powered by AI • Built with React and FastAPI</p>
          <p className="text-sm mt-1">
            Real-time updates via WebSocket • Multimodal AI Analysis
          </p>
        </footer>
        </div>
      </div>
    </ErrorBoundary>
  )
}

export default App

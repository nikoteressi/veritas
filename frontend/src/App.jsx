import React, { useEffect } from 'react'
import UploadForm from './components/UploadForm'
import VerificationResults from './components/VerificationResults'
import WebSocketStatus from './components/WebSocketStatus'
import ErrorBoundary from './components/ErrorBoundary'
import { useWebSocketService } from './hooks/useWebSocketService'
import { useVerificationState } from './hooks/useVerificationState'
import logger from './utils/logger'

function App() {
  // WebSocket connection
  const {
    isConnected,
    sessionId,
    lastMessage,
    sendMessage
  } = useWebSocketService()

  // Verification state management
  const {
    verificationResult,
    isLoading,
    progressData,
    startVerification,
    completeVerification,
    handleWebSocketMessage
  } = useVerificationState()

  // Handle WebSocket messages and delegate to verification state service
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
  }, [lastMessage, handleWebSocketMessage])

  const handleVerificationComplete = (result) => {
    completeVerification(result)
  }

  const handleVerificationStart = () => {
    startVerification()
  }

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
              onVerificationStart={handleVerificationStart}
              onVerificationComplete={handleVerificationComplete}
              isLoading={isLoading}
              sessionId={sessionId}
              isWebSocketConnected={isConnected}
            />
          </div>

          <div className="space-y-6">
            <VerificationResults
              result={verificationResult}
              isLoading={isLoading}
              progressData={progressData}
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

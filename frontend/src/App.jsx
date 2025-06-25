import React, { useState, useEffect } from 'react'
import UploadForm from './components/UploadForm'
import VerificationResults from './components/VerificationResults'
import WebSocketStatus from './components/WebSocketStatus'
import ErrorBoundary from './components/ErrorBoundary'
import { useWebSocket } from './hooks/useWebSocket'

function App() {
  const [verificationResult, setVerificationResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [progressData, setProgressData] = useState(null)

  // WebSocket connection
  const {
    isConnected,
    sessionId,
    lastMessage,
    sendMessage
  } = useWebSocket()

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'progress_update':
          setProgressData(lastMessage.data)
          break
        case 'verification_result':
          setVerificationResult(lastMessage.data)
          setIsLoading(false)
          setProgressData(null)
          break
        case 'error':
          setVerificationResult({
            status: 'error',
            message: lastMessage.data.message
          })
          setIsLoading(false)
          setProgressData(null)
          break
        case 'connection_established':
          console.log('WebSocket connection established:', lastMessage.data)
          break
        case 'pong':
          console.log('WebSocket pong received:', lastMessage.data)
          break
        case 'status_response':
          console.log('WebSocket status response:', lastMessage.data)
          break
        case 'echo':
          console.log('WebSocket echo received:', lastMessage.data)
          break
        default:
          console.log('Unhandled WebSocket message:', lastMessage)
      }
    }
  }, [lastMessage])

  const handleVerificationComplete = (result) => {
    setVerificationResult(result)
    setIsLoading(false)
    setProgressData(null)
  }

  const handleVerificationStart = () => {
    setIsLoading(true)
    setVerificationResult(null)
    setProgressData(null)
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

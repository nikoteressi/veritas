import React from 'react'

function WebSocketStatus({ isConnected, sessionId }) {
  return (
    <div className={`flex items-center space-x-2 text-sm ${
      isConnected ? 'text-green-600' : 'text-red-600'
    }`}>
      <div className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500' : 'bg-red-500'
      }`}></div>
      <span>
        {isConnected ? 'Connected' : 'Disconnected'}
        {sessionId && (
          <span className="text-gray-500 ml-1">
            ({sessionId.slice(0, 8)}...)
          </span>
        )}
      </span>
    </div>
  )
}

export default WebSocketStatus

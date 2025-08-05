import React from 'react'
import { useTranslation } from 'react-i18next'
import { useWebSocketContext } from '../contexts/WebSocketContext'

const WebSocketStatus: React.FC = () => {
  const { t } = useTranslation()
  const { isConnected, sessionId } = useWebSocketContext()

  const getStatusColor = (): string => {
    return isConnected ? 'bg-green-500' : 'bg-red-500'
  }

  const getStatusText = (): string => {
    if (isConnected) {
      return t('websocket.connected')
    }
    return t('websocket.disconnected')
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3" role="status" aria-live="polite">
      <div className="flex items-center space-x-2">
        <div 
          className={`w-2 h-2 rounded-full ${getStatusColor()}`}
          aria-hidden="true"
        ></div>
        <span className="text-sm font-medium text-gray-700">
          {getStatusText()}
        </span>
      </div>
      {sessionId && (
        <div className="mt-1">
          <span className="text-xs text-gray-500">
            {t('websocket.sessionId', { sessionId })}
          </span>
        </div>
      )}
    </div>
  )
}

export default WebSocketStatus
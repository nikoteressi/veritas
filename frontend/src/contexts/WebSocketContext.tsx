import React, { createContext, useContext, ReactNode } from 'react'
import { useWebSocketService } from '../hooks/useWebSocketService'
import { UseWebSocketServiceReturn } from '../types'

interface WebSocketProviderProps {
  children: ReactNode
}

const WebSocketContext = createContext<UseWebSocketServiceReturn | null>(null)

export const useWebSocketContext = (): UseWebSocketServiceReturn => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider')
  }
  return context
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const webSocketState = useWebSocketService()

  return (
    <WebSocketContext.Provider value={webSocketState}>
      {children}
    </WebSocketContext.Provider>
  )
}
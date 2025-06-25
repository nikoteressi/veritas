import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

export function useWebSocket() {
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [lastMessage, setLastMessage] = useState(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5
  const isConnectingRef = useRef(false)
  const socketRef = useRef(null)

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connections
    if (isConnectingRef.current || socketRef.current?.readyState === WebSocket.CONNECTING) {
      console.log('Connection already in progress, skipping...')
      return
    }

    // Close existing connection if any
    if (socketRef.current && socketRef.current.readyState !== WebSocket.CLOSED) {
      console.log('Closing existing connection before creating new one')
      socketRef.current.close(1000, 'Creating new connection')
    }

    try {
      isConnectingRef.current = true
      const ws = new WebSocket(WS_URL)
      socketRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setSocket(ws)
        reconnectAttempts.current = 0
        isConnectingRef.current = false
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('WebSocket message received:', message)

          setLastMessage(message)
          setMessages(prev => [...prev, message])

          // Handle connection establishment
          if (message.type === 'connection_established') {
            setSessionId(message.data.session_id)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setIsConnected(false)
        setSocket(null)
        isConnectingRef.current = false

        // Clear the socket ref if it's the same instance
        if (socketRef.current === ws) {
          socketRef.current = null
        }

        // Attempt to reconnect if not a manual close and not already reconnecting
        if (event.code !== 1000 && event.code !== 1001 && reconnectAttempts.current < maxReconnectAttempts && !reconnectTimeoutRef.current) {
          const timeout = Math.min(Math.pow(2, reconnectAttempts.current) * 1000, 30000) // Exponential backoff with max 30s
          console.log(`Attempting to reconnect in ${timeout}ms... (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`)

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++
            reconnectTimeoutRef.current = null
            connect()
          }, timeout)
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          console.error('Max reconnection attempts reached')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        isConnectingRef.current = false

        // If connection fails immediately, try to reconnect
        if (!isConnected && reconnectAttempts.current < maxReconnectAttempts && !reconnectTimeoutRef.current) {
          const timeout = Math.min(Math.pow(2, reconnectAttempts.current) * 1000, 30000)
          console.log(`Connection failed, retrying in ${timeout}ms...`)

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++
            reconnectTimeoutRef.current = null
            connect()
          }, timeout)
        }
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      isConnectingRef.current = false
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    isConnectingRef.current = false

    if (socketRef.current) {
      socketRef.current.close(1000, 'Manual disconnect')
      socketRef.current = null
    }

    setSocket(null)
    setIsConnected(false)
    setSessionId(null)
  }, [])

  const sendMessage = useCallback((message) => {
    if (socket && isConnected) {
      try {
        const messageString = typeof message === 'string' ? message : JSON.stringify(message)
        socket.send(messageString)
        console.log('WebSocket message sent:', message)
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)
      }
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }, [socket, isConnected])

  const sendPing = useCallback(() => {
    sendMessage({
      type: 'ping',
      timestamp: new Date().toISOString()
    })
  }, [sendMessage])

  const requestStatus = useCallback(() => {
    sendMessage({
      type: 'status_request'
    })
  }, [sendMessage])

  // Auto-connect on mount - use empty dependency array to prevent reconnections in StrictMode
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, []) // Empty dependency array to prevent multiple connections

  // Ping every 30 seconds to keep connection alive
  useEffect(() => {
    if (isConnected) {
      const pingInterval = setInterval(sendPing, 30000)
      return () => clearInterval(pingInterval)
    }
  }, [isConnected, sendPing])

  // Clear messages when disconnected
  useEffect(() => {
    if (!isConnected) {
      setMessages([])
      setLastMessage(null)
    }
  }, [isConnected])

  return {
    socket,
    isConnected,
    sessionId,
    messages,
    lastMessage,
    connect,
    disconnect,
    sendMessage,
    sendPing,
    requestStatus
  }
}

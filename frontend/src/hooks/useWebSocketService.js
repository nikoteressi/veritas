/**
 * Hook for using WebSocket service with React components.
 */
import { useState, useEffect, useRef } from 'react';
import { webSocketService } from '../services/webSocketService';

export const useWebSocketService = () => {
  const [connectionStatus, setConnectionStatus] = useState({
    isConnected: false,
    sessionId: null,
    reconnectAttempts: 0
  });
  const [lastMessage, setLastMessage] = useState(null);
  const unsubscribeRefs = useRef([]);

  // Initialize WebSocket connection
  useEffect(() => {
    const initializeConnection = async () => {
      try {
        // Subscribe to connection events
        const unsubscribeConnection = webSocketService.subscribe('connection_established', () => {
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        const unsubscribeSessionEstablished = webSocketService.subscribe('session_established', () => {
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        const unsubscribeDisconnection = webSocketService.subscribe('connection_lost', () => {
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        const unsubscribeError = webSocketService.subscribe('connection_error', (error) => {
          console.error('WebSocket connection error:', error);
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        // Subscribe to all message types for the lastMessage functionality
        const messageTypes = [
          'progress_update',
          'verification_result',
          'error',
          'connection_established',
          'session_established',
          'pong',
          'status_response',
          'echo'
        ];

        const messageUnsubscribers = messageTypes.map(type => 
          webSocketService.subscribe(type, (data) => {
            setLastMessage({
              type,
              data,
              timestamp: Date.now()
            });
          })
        );

        // Store unsubscribe functions
        unsubscribeRefs.current = [
          unsubscribeConnection,
          unsubscribeSessionEstablished,
          unsubscribeDisconnection,
          unsubscribeError,
          ...messageUnsubscribers
        ];

        // Attempt to connect
        await webSocketService.connect();
        
      } catch (error) {
        console.error('Failed to initialize WebSocket connection:', error);
      }
    };

    initializeConnection();

    // Cleanup on unmount
    return () => {
      unsubscribeRefs.current.forEach(unsubscribe => unsubscribe());
      unsubscribeRefs.current = [];
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  /**
   * Send message via WebSocket.
   * @param {string} type - Message type
   * @param {*} data - Message data
   */
  const sendMessage = (type, data = null) => {
    return webSocketService.send(type, data);
  };

  /**
   * Subscribe to specific message type.
   * @param {string} messageType - Message type to subscribe to
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  const subscribeToMessage = (messageType, callback) => {
    return webSocketService.subscribe(messageType, callback);
  };

  /**
   * Get current connection status.
   * @returns {Object} Connection status
   */
  const getConnectionInfo = () => {
    return webSocketService.getStatus();
  };

  /**
   * Manually reconnect to WebSocket.
   */
  const reconnect = async () => {
    try {
      webSocketService.disconnect();
      await webSocketService.connect();
    } catch (error) {
      console.error('Manual reconnection failed:', error);
    }
  };

  return {
    // Connection status
    isConnected: connectionStatus.isConnected,
    sessionId: connectionStatus.sessionId,
    reconnectAttempts: connectionStatus.reconnectAttempts,
    
    // Message handling
    lastMessage,
    sendMessage,
    subscribeToMessage,
    
    // Connection control
    reconnect,
    getConnectionInfo,
    
    // WebSocket service instance (for advanced usage)
    webSocketService
  };
}; 
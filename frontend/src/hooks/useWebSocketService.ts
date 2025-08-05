/**
 * Hook for using WebSocket service with React components.
 * 
 * WARNING: This hook should typically be used only once in the WebSocketProvider.
 * For most components, use useWebSocketContext() instead to avoid creating
 * multiple WebSocket connections.
 */
import { useState, useEffect, useRef } from 'react';
import { WebSocketState, WebSocketMessage, UseWebSocketServiceReturn } from '../types';
import { webSocketService } from '../services/webSocketService';

export const useWebSocketService = (): UseWebSocketServiceReturn => {
  const [connectionStatus, setConnectionStatus] = useState<{
    isConnected: boolean;
    sessionId: string | null;
    reconnectAttempts: number;
  }>({
    isConnected: false,
    sessionId: null,
    reconnectAttempts: 0
  });
  
  const [wsError, setWsError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const unsubscribeRefs = useRef<(() => void)[]>([]);
  const isInitializing = useRef<boolean>(false);

  // Initialize WebSocket connection and handle cleanup
  useEffect(() => {
    const initializeConnection = async (): Promise<void> => {
      // Prevent multiple initializations (React Strict Mode protection)
      if (isInitializing.current) {
        return;
      }

      // Check if already connected
      const currentStatus = webSocketService.getStatus();
      if (currentStatus.isConnected) {
        setConnectionStatus(currentStatus);
        return;
      }

      isInitializing.current = true;

      try {
        // Subscribe to connection events
        const unsubscribeConnection = webSocketService.subscribe('connection_established', () => {
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        const unsubscribeSessionEstablished = webSocketService.subscribe('session_established', () => {
          const status = webSocketService.getStatus();
          if (import.meta.env.DEV) {
            console.log('WebSocket session established in useWebSocketService:', status);
          }
          setConnectionStatus(status);
        });

        const unsubscribeDisconnection = webSocketService.subscribe('connection_lost', () => {
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        const unsubscribeError = webSocketService.subscribe('connection_error', (error: any) => {
          console.error('WebSocket connection error:', error);
          setWsError(error?.message || 'WebSocket connection error');
          const status = webSocketService.getStatus();
          setConnectionStatus(status);
        });

        // Subscribe to all message types for the lastMessage functionality
        const messageTypes = [
          'verification_result',
          'error',
          'connection_established',
          'session_established',
          'pong',
          'status_response',
          'echo'
          // Progress system messages (steps_definition, step_update) 
          // are handled exclusively by useProgressInterpreter
        ];

        const messageUnsubscribers = messageTypes.map(type => 
          webSocketService.subscribe(type, (data: any) => {
            setLastMessage({
              type: type as any,
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

        // Attempt to connect only if not already connected
        await webSocketService.connect();
        
      } catch (error) {
        console.error('Failed to initialize WebSocket connection:', error);
      } finally {
        isInitializing.current = false;
      }
    };

    initializeConnection();

    // Cleanup on unmount
    return () => {
      // Unsubscribe from all events
      unsubscribeRefs.current.forEach(unsubscribe => unsubscribe());
      unsubscribeRefs.current = [];
      
      // Disconnect WebSocket
      webSocketService.disconnect();
      
      // Reset initialization flag
      isInitializing.current = false;
    };
  }, []);

  /**
   * Send message via WebSocket.
   * @param type - Message type
   * @param data - Message data
   */
  const sendMessage = (type: string, data: any = null): void => {
    webSocketService.send(type, data);
  };

  /**
   * Subscribe to specific message type.
   * @param messageType - Message type to subscribe to
   * @param callback - Callback function
   * @returns Unsubscribe function
   */
  const subscribeToMessage = (messageType: string, callback: (data: any) => void): (() => void) => {
    return webSocketService.subscribe(messageType, callback);
  };

  /**
   * Get current connection status.
   * @returns Connection status
   */
  const getConnectionInfo = (): WebSocketState => {
    const status = webSocketService.getStatus();
    return {
      ...status,
      connectionStatus: status.isConnected ? 'connected' : 'disconnected',
      lastMessage: lastMessage ?? undefined
    };
  };

  /**
   * Manually reconnect to WebSocket.
   */
  const reconnect = async (): Promise<void> => {
    try {
      webSocketService.disconnect();
      await webSocketService.connect();
    } catch (error) {
      console.error('Manual reconnection failed:', error);
      throw error;
    }
  };

  return {
    // Connection status
    isConnected: connectionStatus.isConnected,
    sessionId: connectionStatus.sessionId,
    reconnectAttempts: connectionStatus.reconnectAttempts,
    wsError,
    
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
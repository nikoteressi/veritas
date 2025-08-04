/**
 * Service for managing WebSocket connections and message routing.
 */

import { configurationService } from './configurationService';
import { verificationStateService } from './verificationStateService';
import { errorService, createWebSocketError } from './errorService';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}

interface WebSocketStatus {
  isConnected: boolean;
  sessionId: string | null;
  reconnectAttempts: number;
  readyState: number;
}

interface SessionData {
  session_id: string;
}

interface MessageErrorData {
  error: string;
  rawData: string;
}

type WebSocketEventCallback = (data?: any) => void;

class WebSocketService {
  private socket: WebSocket | null;
  private sessionId: string | null;
  private isConnected: boolean;
  private listeners: Map<string, Set<WebSocketEventCallback>>;
  private reconnectAttempts: number;
  private readonly maxReconnectAttempts: number;
  private readonly reconnectDelay: number;
  private heartbeatInterval: number | null;
  private readonly heartbeatDelay: number;
  
  // Оптимизация подписок
  private eventCache: Map<string, any>;
  private reconnectTimeout: number | null;

  constructor() {
    this.socket = null;
    this.sessionId = null;
    this.isConnected = false;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.heartbeatInterval = null;
    this.heartbeatDelay = 30000; // 30 seconds
    
    // Инициализация оптимизаций
    this.eventCache = new Map();
    this.reconnectTimeout = null;
  }

  /**
   * Connect to WebSocket server.
   */
  async connect(url: string | null = null): Promise<string> {
    // Check if already connected
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return this.sessionId!;
    }

    // Check if connection is in progress
    if (this.socket && this.socket.readyState === WebSocket.CONNECTING) {
      // Wait for existing connection attempt to complete
      return new Promise((resolve, reject) => {
        const checkConnection = () => {
          if (this.socket?.readyState === WebSocket.OPEN && this.sessionId) {
            resolve(this.sessionId);
          } else if (this.socket?.readyState === WebSocket.CLOSED) {
            reject(new Error('Connection failed'));
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
      });
    }

    const wsUrl = url || this._getWebSocketUrl();
    
    return new Promise((resolve, reject) => {
      try {
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this._startHeartbeat();
          this._emit('connection_established');
          if (import.meta.env.DEV) {
            console.log('WebSocket connected');
          }
        };

        this.socket.onmessage = (event) => {
          this._handleMessage(event.data, resolve);
        };

        this.socket.onclose = () => {
          this.isConnected = false;
          this._stopHeartbeat();
          this._emit('connection_lost');
          if (import.meta.env.DEV) {
            console.log('WebSocket connection lost');
          }
          
          // Attempt to reconnect
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this._scheduleReconnect();
          }
        };

        this.socket.onerror = (error) => {
          const wsError = createWebSocketError(
            'WebSocket connection error occurred',
            'connection_error'
          );
          errorService.logError(wsError, { 
            originalError: error,
            url: wsUrl,
            readyState: this.socket?.readyState 
          });
          this._emit('connection_error', wsError);
          reject(wsError);
        };

        // Timeout for connection
        setTimeout(() => {
          if (!this.isConnected) {
            const timeoutError = createWebSocketError(
              'WebSocket connection timeout after 5 seconds',
              'connection_timeout'
            );
            errorService.logError(timeoutError, { url: wsUrl });
            reject(timeoutError);
          }
        }, 5000);

      } catch (error) {
        const connectionError = createWebSocketError(
          'Failed to create WebSocket connection',
          'connection_failed'
        );
        errorService.logError(connectionError, { 
          originalError: error,
          url: wsUrl 
        });
        reject(connectionError);
      }
    });
  }

  /**
   * Disconnect from WebSocket server.
   */
  disconnect(): void {
    if (this.socket) {
      this._stopHeartbeat();
      this.socket.close();
      this.socket = null;
      this.sessionId = null;
      this.isConnected = false;
    }
  }

  /**
   * Send message to WebSocket server.
   */
  send(type: string, data: any = null): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      const notConnectedError = createWebSocketError(
        'WebSocket not connected. Cannot send message',
        'not_connected'
      );
      errorService.logError(notConnectedError, { 
        type, 
        data, 
        readyState: this.socket?.readyState 
      });
      return false;
    }

    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now()
    };

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      const sendError = createWebSocketError(
        'Failed to send WebSocket message',
        'send_failed'
      );
      errorService.logError(sendError, { 
        originalError: error,
        message,
        readyState: this.socket.readyState 
      });
      return false;
    }
  }

  /**
   * Subscribe to WebSocket events.
   */
  subscribe(eventType: string, callback: WebSocketEventCallback): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    
    this.listeners.get(eventType)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const eventListeners = this.listeners.get(eventType);
      if (eventListeners) {
        eventListeners.delete(callback);
        if (eventListeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  /**
   * Get connection status.
   */
  getStatus(): WebSocketStatus {
    return {
      isConnected: this.isConnected,
      sessionId: this.sessionId,
      reconnectAttempts: this.reconnectAttempts,
      readyState: this.socket ? this.socket.readyState : WebSocket.CLOSED
    };
  }

  /**
   * Handle incoming WebSocket messages.
   */
  private _handleMessage(data: string, connectionResolve?: (sessionId: string) => void): void {
    try {
      const message = JSON.parse(data) as WebSocketMessage;
      
      // Handle session establishment
      if (message.type === 'session_established') {
        const sessionData = message.data as SessionData;
        this.sessionId = sessionData.session_id;
        if (connectionResolve) {
          connectionResolve(this.sessionId);
        }
      }
      
      // Handle legacy messages through verification state service
      if (['progress_event', 'verification_result', 'error', 'verification_started'].includes(message.type)) {
        verificationStateService.handleWebSocketMessage(message.type, message.data);
      }
      
      // Emit event to listeners (this handles new progress system messages)
      this._emit(message.type, message.data);
      
    } catch (error) {
      const parseError = createWebSocketError(
        'Failed to parse WebSocket message',
        'parse_failed'
      );
      errorService.logError(parseError, { 
        originalError: error,
        rawData: data,
        dataLength: data.length 
      });
      
      const errorData: MessageErrorData = { 
        error: error instanceof Error ? error.message : 'Unknown error', 
        rawData: data 
      };
      this._emit('message_error', errorData);
    }
  }

  /**
   * Emit event to all listeners with caching optimization.
   */
  private _emit(eventType: string, data?: any): void {
    const eventListeners = this.listeners.get(eventType);
    
    if (eventListeners && eventListeners.size > 0) {
      // Кэшируем последнее событие для оптимизации
      const cacheKey = `${eventType}_last`;
      const lastData = this.eventCache.get(cacheKey);
      
      // Избегаем дублирования одинаковых событий
      if (JSON.stringify(lastData) !== JSON.stringify(data)) {
        this.eventCache.set(cacheKey, data);
        
        // Используем requestAnimationFrame для оптимизации производительности
        requestAnimationFrame(() => {
          eventListeners.forEach(callback => {
            try {
              callback(data);
            } catch (error) {
              const listenerError = createWebSocketError(
                'Error in WebSocket event listener',
                'listener_error'
              );
              errorService.logError(listenerError, { 
                originalError: error,
                eventType,
                data 
              });
            }
          });
        });
      }
    }
  }

  /**
   * Schedule reconnection attempt with debouncing.
   */
  private _scheduleReconnect(): void {
    // Очищаем предыдущий таймер для дебаунсинга
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
      this.reconnectTimeout = null;
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive.
   */
  private _startHeartbeat(): void {
    this._stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected) {
        this.send('ping');
      }
    }, this.heartbeatDelay);
  }

  /**
   * Stop heartbeat.
   */
  private _stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Get WebSocket URL from configuration service.
   */
  private _getWebSocketUrl(): string {
    return configurationService.webSocket.url;
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();
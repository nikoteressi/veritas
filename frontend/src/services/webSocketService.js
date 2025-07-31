/**
 * Service for managing WebSocket connections and message routing.
 */

import { configurationService } from './configurationService.js';

class WebSocketService {
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
  }

  /**
   * Connect to WebSocket server.
   * @param {string} url - WebSocket URL
   * @returns {Promise<string>} Session ID
   */
  async connect(url = null) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return this.sessionId;
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
          console.log('WebSocket connected');
        };

        this.socket.onmessage = (event) => {
          this._handleMessage(event.data, resolve);
        };

        this.socket.onclose = () => {
          this.isConnected = false;
          this._stopHeartbeat();
          this._emit('connection_lost');
          console.log('WebSocket disconnected');
          
          // Attempt to reconnect
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this._scheduleReconnect();
          }
        };

        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this._emit('connection_error', error);
          reject(error);
        };

        // Timeout for connection
        setTimeout(() => {
          if (!this.isConnected) {
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);

      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server.
   */
  disconnect() {
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
   * @param {string} type - Message type
   * @param {*} data - Message data
   */
  send(type, data = null) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected. Cannot send message:', { type, data });
      return false;
    }

    const message = {
      type,
      data,
      timestamp: Date.now()
    };

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      return false;
    }
  }

  /**
   * Subscribe to WebSocket events.
   * @param {string} eventType - Event type to listen for
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  subscribe(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    
    this.listeners.get(eventType).add(callback);
    
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
   * @returns {Object} Connection status information
   */
  getStatus() {
    return {
      isConnected: this.isConnected,
      sessionId: this.sessionId,
      reconnectAttempts: this.reconnectAttempts,
      readyState: this.socket ? this.socket.readyState : WebSocket.CLOSED
    };
  }

  /**
   * Handle incoming WebSocket messages.
   * @private
   */
  _handleMessage(data, connectionResolve = null) {
    try {
      const message = JSON.parse(data);
      
      // Handle session establishment
      if (message.type === 'session_established') {
        this.sessionId = message.data.session_id;
        if (connectionResolve) {
          connectionResolve(this.sessionId);
        }
      }
      
      // Handle new progress system messages
      if (['steps_definition', 'progress_update', 'step_update'].includes(message.type)) {
        // Route progress messages to verification state service
        import('./verificationStateService.js').then(({ verificationStateService }) => {
          verificationStateService.handleWebSocketMessage(message.type, message.data);
        });
      }
      
      // Emit event to listeners
      this._emit(message.type, message.data);
      
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      this._emit('message_error', { error: error.message, rawData: data });
    }
  }

  /**
   * Emit event to listeners.
   * @private
   */
  _emit(eventType, data = null) {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in WebSocket event listener:', error);
        }
      });
    }
  }

  /**
   * Schedule reconnection attempt.
   * @private
   */
  _scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive.
   * @private
   */
  _startHeartbeat() {
    this._stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected) {
        this.send('ping');
      }
    }, this.heartbeatDelay);
  }

  /**
   * Stop heartbeat.
   * @private
   */
  _stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Get WebSocket URL from configuration service.
   * @private
   */
  _getWebSocketUrl() {
    return configurationService.webSocket.url;
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();
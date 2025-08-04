/**
 * Service for managing verification state and coordinating between components.
 */
import { VerificationResult } from '../types';

interface VerificationState {
  verificationResult: VerificationResult | null;
  isLoading: boolean;
  error: string | null;
}

interface VerificationOptions {
  [key: string]: any;
}

interface VerificationSummary {
  hasResult: boolean;
  isProcessing: boolean;
  hasError: boolean;
  status: 'idle' | 'loading' | 'completed' | 'error';
}

interface ProgressEventData {
  event_name: string;
  payload?: any;
}

interface WebSocketMessageData {
  message?: string;
  [key: string]: any;
}

type StateListener = (newState: VerificationState, previousState: VerificationState) => void;
type EventType = 'verification_started' | 'verification_completed' | 'verification_error' | 'state_cleared' | 'state_reset' | 'progress_event';

class VerificationStateService {
  private listeners: Map<symbol, StateListener>;
  private state: VerificationState;

  constructor() {
    this.listeners = new Map();
    this.state = {
      verificationResult: null,
      isLoading: false,
      error: null
    };
  }

  /**
   * Get current verification state.
   */
  getState(): VerificationState {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes.
   */
  subscribe(callback: StateListener): () => void {
    const listenerId = Symbol('listener');
    this.listeners.set(listenerId, callback);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listenerId);
    };
  }

  /**
   * Start a verification process.
   */
  startVerification(options: VerificationOptions = {}): void {
    this.updateState({
      isLoading: true,
      verificationResult: null,
      error: null
    });

    this._emit('verification_started', options);
  }

  /**
   * Complete verification with result.
   */
  completeVerification(result: VerificationResult): void {
    this.updateState({
      verificationResult: result,
      isLoading: false,
      error: null
    });

    this._emit('verification_completed', result);
  }

  /**
   * Set verification error.
   */
  setError(error: string | Error): void {
    const errorMessage = error instanceof Error ? error.message : error;
    
    this.updateState({
      error: errorMessage,
      isLoading: false
    });

    this._emit('verification_error', errorMessage);
  }

  /**
   * Clear verification state.
   */
  clearState(): void {
    this.updateState({
      verificationResult: null,
      isLoading: false,
      error: null
    });

    this._emit('state_cleared');
  }

  /**
   * Reset verification state to initial state.
   */
  reset(): void {
    this.clearState();
    this._emit('state_reset');
  }

  /**
   * Update state and notify listeners.
   */
  private updateState(updates: Partial<VerificationState>): void {
    const previousState = { ...this.state };
    this.state = { ...this.state, ...updates };
    
    // Notify all listeners
    this.listeners.forEach(callback => {
      try {
        callback(this.state, previousState);
      } catch (error) {
        console.error('Error in state listener:', error);
      }
    });
  }

  /**
   * Emit event to listeners.
   */
  private _emit(eventType: EventType, data: any = null): void {
    // This could be extended to support event-specific listeners
    // For now, all listeners receive state updates
    
    console.debug(`Verification state event: ${eventType}`, data);
  }

  /**
   * Handle WebSocket message for verification updates.
   */
  handleWebSocketMessage(messageType: string, data: WebSocketMessageData): void {
    switch (messageType) {
      case 'progress_event':
        this.handleProgressEvent(data as ProgressEventData);
        break;
      
      case 'verification_result':
        this.completeVerification(data as VerificationResult);
        break;
      
      case 'error':
        this.setError(data.message || 'Verification failed');
        break;
      
      case 'verification_started':
        // Handle verification started confirmation
        break;
      
      default:
        console.warn('Unhandled WebSocket message in verification state:', messageType, data);
    }
  }

  /**
   * Handle progress event from the new event-driven system.
   */
  handleProgressEvent(eventData: ProgressEventData): void {
    if (!eventData || !eventData.event_name) {
      console.warn('Invalid progress event data:', eventData);
      return;
    }

    this._emit('progress_event', eventData);
  }

  /**
   * Get verification summary for display.
   */
  getVerificationSummary(): VerificationSummary {
    const { verificationResult, isLoading, error } = this.state;
    
    return {
      hasResult: !!verificationResult,
      isProcessing: isLoading,
      hasError: !!error,
      status: this._getStatus()
    };
  }

  /**
   * Get current verification status.
   */
  private _getStatus(): 'idle' | 'loading' | 'completed' | 'error' {
    if (this.state.error) return 'error';
    if (this.state.isLoading) return 'loading';
    if (this.state.verificationResult) return 'completed';
    return 'idle';
  }
}

// Export singleton instance
export const verificationStateService = new VerificationStateService();
/**
 * Service for managing verification state and coordinating between components.
 */
import logger from '../utils/logger';

class VerificationStateService {
  constructor() {
    this.listeners = new Map();
    this.state = {
      verificationResult: null,
      isLoading: false,
      progressData: null,
      error: null
    };
  }

  /**
   * Get current verification state.
   * @returns {Object} Current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes.
   * @param {Function} callback - Callback function that receives the new state
   * @returns {Function} Unsubscribe function
   */
  subscribe(callback) {
    const listenerId = Symbol('listener');
    this.listeners.set(listenerId, callback);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listenerId);
    };
  }

  /**
   * Start a verification process.
   * @param {Object} options - Verification options
   */
  startVerification(options = {}) {
    this.updateState({
      isLoading: true,
      verificationResult: null,
      progressData: null,
      error: null
    });

    this._emit('verification_started', options);
  }

  /**
   * Update verification progress.
   * @param {Object} progressData - Progress information
   */
  updateProgress(progressData) {
    // Prevent duplicate progress updates
    const currentProgress = this.state.progressData;
    if (currentProgress && 
        currentProgress.step === progressData.step && 
        currentProgress.progress === progressData.progress) {
      return; // Skip duplicate update
    }

    this.updateState({
      progressData
    });

    this._emit('progress_updated', progressData);
  }

  /**
   * Complete verification with result.
   * @param {Object} result - Verification result
   */
  completeVerification(result) {
    this.updateState({
      verificationResult: result,
      isLoading: false,
      progressData: null,
      error: null
    });

    this._emit('verification_completed', result);
  }

  /**
   * Set verification error.
   * @param {string|Error} error - Error message or Error object
   */
  setError(error) {
    const errorMessage = error instanceof Error ? error.message : error;
    
    this.updateState({
      error: errorMessage,
      isLoading: false,
      progressData: null
    });

    this._emit('verification_error', errorMessage);
  }

  /**
   * Clear verification state.
   */
  clearState() {
    this.updateState({
      verificationResult: null,
      isLoading: false,
      progressData: null,
      error: null
    });

    this._emit('state_cleared');
  }

  /**
   * Reset verification state to initial state.
   */
  reset() {
    this.clearState();
    this._emit('state_reset');
  }

  /**
   * Update state and notify listeners.
   * @private
   */
  updateState(updates) {
    const previousState = { ...this.state };
    this.state = { ...this.state, ...updates };
    
    // Notify all listeners
    this.listeners.forEach(callback => {
      try {
        callback(this.state, previousState);
      } catch (error) {
        logger.error('Error in state listener:', error);
      }
    });
  }

  /**
   * Emit event to listeners.
   * @private
   */
  _emit(eventType, data = null) {
    // This could be extended to support event-specific listeners
    // For now, all listeners receive state updates
    
    // Only log important events, not progress updates
    if (eventType !== 'progress_updated') {
      logger.debug(`Verification state event: ${eventType}`, data);
    }
  }

  /**
   * Handle WebSocket message for verification updates.
   * @param {string} messageType - Type of WebSocket message
   * @param {*} data - Message data
   */
  handleWebSocketMessage(messageType, data) {
    switch (messageType) {
      case 'progress_update':
        this.updateProgress(data);
        break;
      
      case 'verification_result':
        this.completeVerification(data);
        break;
      
      case 'error':
        this.setError(data.message || 'Verification failed');
        break;
      
      case 'verification_started':
        // Handle verification started confirmation
        break;
      
      default:
        logger.warn('Unhandled WebSocket message in verification state:', messageType, data);
    }
  }

  /**
   * Get verification summary for display.
   * @returns {Object} Verification summary
   */
  getVerificationSummary() {
    const { verificationResult, isLoading, progressData, error } = this.state;
    
    return {
      hasResult: !!verificationResult,
      isProcessing: isLoading,
      hasError: !!error,
      progressStep: progressData?.step || null,
      progressMessage: progressData?.message || null,
      status: this._getStatus()
    };
  }

  /**
   * Get current verification status.
   * @private
   */
  _getStatus() {
    if (this.state.error) return 'error';
    if (this.state.isLoading) return 'loading';
    if (this.state.verificationResult) return 'completed';
    return 'idle';
  }
}

// Export singleton instance
export const verificationStateService = new VerificationStateService(); 
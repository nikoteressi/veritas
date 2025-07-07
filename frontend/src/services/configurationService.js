/**
 * Configuration service for frontend application settings.
 */

class ConfigurationService {
  constructor() {
    // File upload configuration
    this.fileUpload = {
      maxSize: 10 * 1024 * 1024, // 10MB
      allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'],
      allowedExtensions: ['jpg', 'jpeg', 'png', 'gif', 'webp']
    };

    // Validation configuration
    this.validation = {
      minPromptLength: 10,
      maxPromptLength: 2000,
      sessionIdMinLength: 8,
      sessionIdMaxLength: 100
    };

    // UI configuration
    this.ui = {
      progressUpdateInterval: 100, // ms
      connectionTimeout: 5000, // 5 seconds
      maxReconnectAttempts: 5,
      reconnectDelay: 1000, // 1 second
      heartbeatInterval: 30000, // 30 seconds
      debounceDelay: 300 // ms
    };

    // API configuration
    this.api = {
      baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
      timeout: 60000, // 60 seconds
      retryAttempts: 3,
      retryDelay: 1000 // 1 second
    };

    // WebSocket configuration
    this.webSocket = {
      url: this._getWebSocketUrl(),
      reconnectAttempts: 5,
      reconnectDelay: 1000,
      heartbeatInterval: 30000,
      connectionTimeout: 5000
    };

    // Verification steps
    this.verificationSteps = [
      { key: 'validation', label: 'Validation', message: 'Validating request...' },
      { key: 'image_analysis', label: 'Image Analysis', message: 'Analyzing image content...' },
      { key: 'reputation_retrieval', label: 'Reputation Check', message: 'Checking user reputation...' },
      { key: 'temporal_analysis', label: 'Temporal Analysis', message: 'Analyzing temporal context...' },
      { key: 'fact_checking', label: 'Fact Checking', message: 'Fact-checking claims...' },
      { key: 'verdict_generation', label: 'Verdict Generation', message: 'Generating verdict...' },
      { key: 'motives_analysis', label: 'Motives Analysis', message: 'Analyzing user motives...' },
      { key: 'reputation_update', label: 'Reputation Update', message: 'Updating reputation...' },
      { key: 'result_storage', label: 'Saving Results', message: 'Saving results...' }
    ];

    // Load environment overrides
    this._loadFromEnvironment();
  }

  /**
   * Validate file against size and type constraints.
   * @param {File} file - File to validate
   * @throws {Error} If validation fails
   */
  validateFile(file) {
    if (!file) {
      throw new Error('No file provided');
    }

    if (file.size > this.fileUpload.maxSize) {
      throw new Error(`File too large. Maximum size is ${this.fileUpload.maxSize / (1024 * 1024)}MB`);
    }

    if (!this.fileUpload.allowedTypes.includes(file.type)) {
      throw new Error(`Invalid file type. Allowed types: ${this.fileUpload.allowedTypes.join(', ')}`);
    }

    return true;
  }

  /**
   * Validate prompt text.
   * @param {string} prompt - Prompt to validate
   * @throws {Error} If validation fails
   */
  validatePrompt(prompt) {
    if (!prompt || !prompt.trim()) {
      throw new Error('Prompt is required');
    }

    const cleanPrompt = prompt.trim();

    if (cleanPrompt.length < this.validation.minPromptLength) {
      throw new Error(`Prompt too short. Minimum length is ${this.validation.minPromptLength} characters`);
    }

    if (cleanPrompt.length > this.validation.maxPromptLength) {
      throw new Error(`Prompt too long. Maximum length is ${this.validation.maxPromptLength} characters`);
    }

    return cleanPrompt;
  }

  /**
   * Get step information by key.
   * @param {string} stepKey - Step key
   * @returns {Object|null} Step information
   */
  getStep(stepKey) {
    return this.verificationSteps.find(step => step.key === stepKey) || null;
  }

  /**
   * Get step label by key.
   * @param {string} stepKey - Step key
   * @returns {string} Step label
   */
  getStepLabel(stepKey) {
    const step = this.getStep(stepKey);
    return step ? step.label : stepKey.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  /**
   * Get step message by key.
   * @param {string} stepKey - Step key
   * @returns {string} Step message
   */
  getStepMessage(stepKey) {
    const step = this.getStep(stepKey);
    return step ? step.message : `Processing ${this.getStepLabel(stepKey)}...`;
  }

  /**
   * Check if file type is supported.
   * @param {string} type - MIME type
   * @returns {boolean} Whether type is supported
   */
  isSupportedFileType(type) {
    return this.fileUpload.allowedTypes.includes(type);
  }

  /**
   * Format file size for display.
   * @param {number} bytes - File size in bytes
   * @returns {string} Formatted file size
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Get application version.
   * @returns {string} Application version
   */
  getVersion() {
    return import.meta.env.VITE_APP_VERSION || '1.0.0';
  }

  /**
   * Get application environment.
   * @returns {string} Application environment
   */
  getEnvironment() {
    return import.meta.env.MODE || 'development';
  }

  /**
   * Check if in development mode.
   * @returns {boolean} Whether in development mode
   */
  isDevelopment() {
    return this.getEnvironment() === 'development';
  }

  /**
   * Check if in production mode.
   * @returns {boolean} Whether in production mode
   */
  isProduction() {
    return this.getEnvironment() === 'production';
  }

  /**
   * Get all configuration as object.
   * @returns {Object} All configuration
   */
  getAllConfig() {
    return {
      fileUpload: this.fileUpload,
      validation: this.validation,
      ui: this.ui,
      api: this.api,
      webSocket: this.webSocket,
      verificationSteps: this.verificationSteps
    };
  }

  /**
   * Load configuration overrides from environment variables.
   * @private
   */
  _loadFromEnvironment() {
    // API configuration
    if (import.meta.env.VITE_API_TIMEOUT) {
      this.api.timeout = parseInt(import.meta.env.VITE_API_TIMEOUT);
    }

    // File upload configuration
    if (import.meta.env.VITE_MAX_FILE_SIZE) {
      this.fileUpload.maxSize = parseInt(import.meta.env.VITE_MAX_FILE_SIZE);
    }

    // WebSocket configuration
    if (import.meta.env.VITE_WS_URL) {
      this.webSocket.url = import.meta.env.VITE_WS_URL;
    }

    // UI configuration
    if (import.meta.env.VITE_CONNECTION_TIMEOUT) {
      this.ui.connectionTimeout = parseInt(import.meta.env.VITE_CONNECTION_TIMEOUT);
    }
  }

  /**
   * Get WebSocket URL based on API base URL.
   * @private
   */
  _getWebSocketUrl() {
    // Use environment variable if set, otherwise derive from API base URL
    if (import.meta.env.VITE_WS_URL) {
      return import.meta.env.VITE_WS_URL;
    }
    
    // Derive WebSocket URL from API base URL
    const apiUrl = new URL(this.api.baseUrl);
    const protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${apiUrl.host}/ws`;
  }
}

// Export singleton instance
export const configurationService = new ConfigurationService(); 
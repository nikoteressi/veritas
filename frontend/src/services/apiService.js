/**
 * Service for handling API requests and communication.
 */
import axios from 'axios';
import { retryWithBackoff } from '../utils/errorHandling';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiService = {
  /**
   * Submit a verification request to the API.
   * @param {Object} options - Request options
   * @param {File} options.file - The image file to verify
   * @param {string} options.prompt - User prompt/question
   * @param {string} options.sessionId - Optional WebSocket session ID
   * @returns {Promise} API response
   */
  async submitVerificationRequest({ file, prompt, sessionId }) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt);

    // Add session ID if provided for real-time updates
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const makeRequest = async () => {
      return await axios.post(`${API_BASE_URL}/verify-post`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout
      });
    };

    // Use retry logic for resilience
    return await retryWithBackoff(makeRequest, 3, 1000);
  },

  /**
   * Get the status of a verification request.
   * @param {string} verificationId - The verification ID
   * @returns {Promise} API response
   */
  async getVerificationStatus(verificationId) {
    const response = await axios.get(`${API_BASE_URL}/verification-status/${verificationId}`);
    return response.data;
  },

  /**
   * Get user reputation information.
   * @param {string} nickname - User nickname
   * @returns {Promise} API response
   */
  async getUserReputation(nickname) {
    const response = await axios.get(`${API_BASE_URL}/reputation/${nickname}`);
    return response.data;
  },

  /**
   * Health check endpoint.
   * @returns {Promise} API response
   */
  async healthCheck() {
    const response = await axios.get(`${API_BASE_URL.replace('/api/v1', '')}/health`);
    return response.data;
  }
}; 
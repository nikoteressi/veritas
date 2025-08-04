/**
 * Service for handling API requests and communication.
 */
import axios, { AxiosResponse } from 'axios'
import { retryWithBackoff } from '../utils/errorHandling'
import { VerificationResult, ReputationData, HealthCheckResponse } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

interface VerificationRequestOptions {
  file: File
  prompt: string
  sessionId?: string | null
}

interface ApiService {
  submitVerificationRequest(options: VerificationRequestOptions): Promise<AxiosResponse<VerificationResult>>
  getVerificationStatus(verificationId: string): Promise<VerificationResult>
  getUserReputation(nickname: string): Promise<ReputationData>
  healthCheck(): Promise<HealthCheckResponse>
}

export const apiService: ApiService = {
  /**
   * Submit a verification request to the API.
   */
  async submitVerificationRequest({ file, prompt, sessionId }: VerificationRequestOptions): Promise<AxiosResponse<VerificationResult>> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('prompt', prompt)

    // Add session ID if provided for real-time updates
    if (sessionId) {
      formData.append('session_id', sessionId)
    } else {
      console.log('No sessionId provided - WebSocket updates will not work')
    }

    const makeRequest = async (): Promise<AxiosResponse<VerificationResult>> => {
      return await axios.post(`${API_BASE_URL}/verify-post`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout
      })
    }

    // Use retry logic for resilience
    const response = await retryWithBackoff(makeRequest, 3, 1000)
    return response
  },

  /**
   * Get the status of a verification request.
   */
  async getVerificationStatus(verificationId: string): Promise<VerificationResult> {
    const response = await axios.get(`${API_BASE_URL}/verification-status/${verificationId}`)
    return response.data
  },

  /**
   * Get user reputation information.
   */
  async getUserReputation(nickname: string): Promise<ReputationData> {
    const response = await axios.get(`${API_BASE_URL}/reputation/${nickname}`)
    return response.data
  },

  /**
   * Health check endpoint.
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await axios.get(`${API_BASE_URL.replace('/api/v1', '')}/health`)
    return response.data
  }
}
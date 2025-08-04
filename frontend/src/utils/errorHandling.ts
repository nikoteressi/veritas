/**
 * Error handling utilities for the frontend
 */
import { configurationService } from '../services/configurationService'

export class APIError extends Error {
  public readonly status: number
  public readonly code: string
  public readonly details: Record<string, any>

  constructor(message: string, status: number, code: string, details: Record<string, any> = {}) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.code = code
    this.details = details
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ValidationError extends Error {
  public readonly field: string | null

  constructor(message: string, field: string | null = null) {
    super(message)
    this.name = 'ValidationError'
    this.field = field
  }
}

interface ErrorResponse {
  message?: string
  detail?: string
  error_code?: string
  details?: Record<string, any>
}

interface AxiosError {
  response?: {
    status: number
    data?: ErrorResponse
  }
  request?: any
  message?: string
}

interface NotificationOptions {
  type: 'error' | 'success' | 'warning' | 'info'
  message: string
  duration?: number
}

type NotificationFunction = (options: NotificationOptions) => void

/**
 * Handle API response errors
 */
export const handleAPIError = (error: AxiosError): never => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response
    const message = data?.message || data?.detail || 'An error occurred'
    const code = data?.error_code || 'UNKNOWN_ERROR'
    const details = data?.details || {}
    
    throw new APIError(message, status, code, details)
  } else if (error.request) {
    // Network error
    throw new NetworkError('Network error - please check your connection')
  } else {
    // Other error
    throw new Error(error.message || 'An unexpected error occurred')
  }
}

/**
 * Get user-friendly error message
 */
export const getErrorMessage = (error: Error): string => {
  if (error instanceof APIError) {
    switch (error.code) {
      case 'FILE_TOO_LARGE':
        return 'The image file is too large. Please use an image smaller than 10MB.'
      case 'INVALID_FILE_EXTENSION':
        return 'Invalid file type. Please upload a JPEG, PNG, GIF, or WebP image.'
      case 'PROMPT_TOO_SHORT':
        return 'Please enter a longer question or prompt (at least 5 characters).'
      case 'PROMPT_TOO_LONG':
        return 'Your prompt is too long. Please keep it under 1000 characters.'
      case 'LLM_SERVICE_ERROR':
        return 'The AI analysis service is temporarily unavailable. Please try again later.'
      case 'DATABASE_ERROR':
        return 'Database service is temporarily unavailable. Please try again later.'
      case 'SERVICE_UNAVAILABLE':
        return 'Some services are temporarily unavailable. The analysis may be limited.'
      default:
        return error.message
    }
  } else if (error instanceof NetworkError) {
    return 'Network connection error. Please check your internet connection and try again.'
  } else if (error instanceof ValidationError) {
    return error.message
  } else {
    return error.message || 'An unexpected error occurred'
  }
}

/**
 * Validate file before upload
 */
export const validateFile = (file: File): boolean => {
  try {
    // Delegate to configuration service for validation
    return configurationService.validateFile(file)
  } catch (error) {
    // Convert configuration service errors to ValidationError
    throw new ValidationError((error as Error).message)
  }
}

/**
 * Validate prompt text
 */
export const validatePrompt = (prompt: string): string => {
  try {
    // Delegate to configuration service for validation
    return configurationService.validatePrompt(prompt)
  } catch (error) {
    // Convert configuration service errors to ValidationError
    throw new ValidationError((error as Error).message)
  }
}

/**
 * Retry function with exponential backoff
 */
export const retryWithBackoff = async <T>(
  fn: () => Promise<T>, 
  maxRetries: number = 3, 
  baseDelay: number = 1000
): Promise<T> => {
  let lastError: Error

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error as Error
      
      // Don't retry on validation errors or client errors
      if (error instanceof ValidationError || 
          (error instanceof APIError && error.status >= 400 && error.status < 500)) {
        throw error
      }

      if (attempt < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, attempt)
        await new Promise(resolve => setTimeout(resolve, delay))
      }
    }
  }

  throw lastError!
}

/**
 * Log error for debugging
 */
export const logError = (error: Error, context: Record<string, any> = {}): void => {
  console.error('Error occurred:', {
    message: error.message,
    name: error.name,
    stack: error.stack,
    context,
    timestamp: new Date().toISOString()
  })
}

/**
 * Show user-friendly error notification
 */
export const showErrorNotification = (error: Error, notificationFn?: NotificationFunction): void => {
  const message = getErrorMessage(error)
  logError(error)
  
  if (notificationFn) {
    notificationFn({
      type: 'error',
      message,
      duration: 5000
    })
  } else {
    // Fallback to alert if no notification function provided
    alert(message)
  }
}
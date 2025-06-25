/**
 * Error handling utilities for the frontend
 */

export class APIError extends Error {
  constructor(message, status, code, details = {}) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.code = code
    this.details = details
  }
}

export class NetworkError extends Error {
  constructor(message) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ValidationError extends Error {
  constructor(message, field = null) {
    super(message)
    this.name = 'ValidationError'
    this.field = field
  }
}

/**
 * Handle API response errors
 */
export const handleAPIError = (error) => {
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
export const getErrorMessage = (error) => {
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
export const validateFile = (file) => {
  if (!file) {
    throw new ValidationError('Please select a file')
  }

  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
  if (!allowedTypes.includes(file.type)) {
    throw new ValidationError('Please select a valid image file (JPEG, PNG, GIF, or WebP)')
  }

  const maxSize = 10 * 1024 * 1024 // 10MB
  if (file.size > maxSize) {
    throw new ValidationError('File size must be less than 10MB')
  }

  const minSize = 1024 // 1KB
  if (file.size < minSize) {
    throw new ValidationError('File is too small. Please select a valid image.')
  }
}

/**
 * Validate prompt text
 */
export const validatePrompt = (prompt) => {
  if (!prompt || !prompt.trim()) {
    throw new ValidationError('Please enter a question or prompt')
  }

  const trimmed = prompt.trim()
  
  if (trimmed.length < 5) {
    throw new ValidationError('Please enter a longer question (at least 5 characters)')
  }

  if (trimmed.length > 1000) {
    throw new ValidationError('Please keep your question under 1000 characters')
  }

  // Check for potentially harmful content
  const suspiciousPatterns = [
    /<script[^>]*>.*?<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi
  ]

  for (const pattern of suspiciousPatterns) {
    if (pattern.test(trimmed)) {
      throw new ValidationError('Please remove any code or scripts from your prompt')
    }
  }

  return trimmed
}

/**
 * Retry function with exponential backoff
 */
export const retryWithBackoff = async (fn, maxRetries = 3, baseDelay = 1000) => {
  let lastError

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      
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

  throw lastError
}

/**
 * Log error for debugging
 */
export const logError = (error, context = {}) => {
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
export const showErrorNotification = (error, notificationFn) => {
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

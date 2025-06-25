import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import {
  validateFile,
  validatePrompt,
  handleAPIError,
  retryWithBackoff,
  showErrorNotification
} from '../utils/errorHandling'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

function UploadForm({
  onVerificationStart,
  onVerificationComplete,
  isLoading,
  sessionId,
  isWebSocketConnected
}) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState(null)
  const [validationErrors, setValidationErrors] = useState({})

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null)
    setValidationErrors({})

    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      setError(`File rejected: ${rejection.errors[0]?.message || 'Invalid file'}`)
      return
    }

    const file = acceptedFiles[0]
    if (file) {
      try {
        validateFile(file)
        setSelectedFile(file)

        // Create preview URL
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
      } catch (validationError) {
        setError(validationError.message)
      }
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  })

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setValidationErrors({})

    try {
      // Validate inputs
      if (!selectedFile) {
        setValidationErrors({ file: 'Please select an image file' })
        return
      }

      validateFile(selectedFile)
      const validatedPrompt = validatePrompt(prompt)

      onVerificationStart()

      // Create verification request with retry logic
      const makeRequest = async () => {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('prompt', validatedPrompt)

        // Add session ID if WebSocket is connected for real-time updates
        if (isWebSocketConnected && sessionId) {
          formData.append('session_id', sessionId)
        }

        return await axios.post(`${API_BASE_URL}/verify-post`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 60000, // 60 second timeout
        })
      }

      const response = await retryWithBackoff(makeRequest, 3, 1000)

      // If using WebSocket, the result will come via WebSocket
      // Otherwise, handle the response directly
      if (!isWebSocketConnected || !sessionId) {
        onVerificationComplete(response.data)
      }
      // If WebSocket is connected, the result will be handled by the WebSocket message handler

    } catch (error) {
      console.error('Verification error:', error)

      try {
        handleAPIError(error)
      } catch (handledError) {
        const errorMessage = handledError.message || 'An error occurred during verification'
        setError(errorMessage)

        onVerificationComplete({
          status: 'error',
          message: errorMessage
        })
      }
    }
  }

  const clearFile = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Upload Social Media Post
      </h2>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-red-500 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Area */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Screenshot Image
          </label>
          
          {!selectedFile ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-400 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              <div className="space-y-2">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="text-gray-600">
                  {isDragActive
                    ? 'Drop the image here...'
                    : 'Drag & drop an image here, or click to select'}
                </p>
                <p className="text-sm text-gray-500">
                  PNG, JPG, GIF up to 10MB
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-w-full h-48 object-contain mx-auto rounded-lg border"
                />
                <button
                  type="button"
                  onClick={clearFile}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="text-sm text-gray-600 text-center">
                {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            </div>
          )}
        </div>

        {/* Prompt Input */}
        <div>
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
            Your Question or Prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="What would you like to verify about this post? (e.g., 'Is this claim about climate change accurate?')"
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-gray-900 placeholder-gray-500 bg-white"
            disabled={isLoading}
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!selectedFile || !prompt.trim() || isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Verifying...' : 'Verify Post'}
        </button>
      </form>
    </div>
  )
}

export default UploadForm

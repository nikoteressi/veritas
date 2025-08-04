import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useDropzone } from 'react-dropzone'
import { useFileUpload } from '../hooks/useFileUpload'
import { useVerification } from '../hooks/useVerification'
import { useVerificationContext } from '../contexts/VerificationContext'
import { useWebSocketContext } from '../contexts/WebSocketContext'
import { configurationService } from '../services/configurationService'
import Button from './ui/Button'
import ErrorBoundary from './ErrorBoundary'
import ErrorDisplay from './ui/ErrorDisplay'

function UploadForm() {
  const { t } = useTranslation()
  const { isLoading, onVerificationStart, onVerificationComplete } = useVerificationContext()
  const { sessionId, isConnected: isWebSocketConnected } = useWebSocketContext()
  const [prompt, setPrompt] = useState('')
  
  // Use custom hooks for file handling and verification
  const {
    selectedFile,
    previewUrl,
    fileError,
    handleDropzoneFiles,
    clearFile: clearSelectedFile,
    cleanup
  } = useFileUpload()
  
  const {
    error: verificationError,
    submitVerification
  } = useVerification()
  
  // Combine errors for display
  const error = fileError || verificationError
  
  // Cleanup preview URL on unmount
  useEffect(() => {
    return cleanup
  }, [cleanup])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDropzoneFiles,
    accept: {
      'image/*': configurationService.fileUpload.allowedExtensions.map(ext => `.${ext}`)
    },
    multiple: false,
    maxSize: configurationService.fileUpload.maxSize
  })

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    
    await submitVerification(
      selectedFile,
      prompt,
      sessionId,
      isWebSocketConnected,
      onVerificationStart,
      onVerificationComplete
    )
  }

  const handleClearFile = () => {
    clearSelectedFile()
  }

  return (
    <ErrorBoundary level="component" showDetails={import.meta.env.DEV}>
      <div className="max-w-3xl mx-auto">
        {/* Error Display */}
        {error && (
          <ErrorDisplay 
            error={error}
            onDismiss={() => {
              // Clear both file and verification errors
              if (fileError) clearSelectedFile();
              // Note: verificationError would need to be cleared by the verification hook
            }}
            showDetails={import.meta.env.DEV}
            className="mb-8"
            size="md"
          />
        )}

        <form onSubmit={handleSubmit} className="space-y-10">
        {/* Enhanced File Upload Area */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 mb-4">
            {t('upload.fileUpload.label')}
          </label>
          
          {!selectedFile ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-2xl p-20 text-center cursor-pointer transition-all duration-300 ${
                isDragActive
                  ? 'border-blue-400 bg-blue-50 scale-[1.02] shadow-lg'
                  : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50 hover:shadow-md'
              }`}
            >
              <input {...getInputProps()} />
              <div className="space-y-6">
                {/* Enhanced Upload Icon */}
                <div className="mx-auto w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center">
                  <svg
                    className="w-10 h-10 text-blue-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                
                <div className="space-y-3">
                  <p className="text-2xl font-semibold text-gray-900">
                    {isDragActive ? t('upload.fileUpload.dropActive') : t('upload.fileUpload.title')}
                  </p>
                  <p className="text-lg text-gray-600">
                    {t('upload.fileUpload.description')}
                  </p>
                  <p className="text-sm text-gray-500">
                    {t('upload.fileUpload.supportedFormats')}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative bg-white border-2 border-gray-200 rounded-2xl p-8 shadow-sm">
                <img
                  src={previewUrl || ''}
                  alt={t('upload.fileUpload.previewAlt')}
                  className="max-w-full h-80 object-contain mx-auto rounded-lg shadow-md"
                />
                <button
                  type="button"
                  onClick={handleClearFile}
                  className="absolute top-4 right-4 bg-red-500 text-white rounded-full p-2.5 hover:bg-red-600 transition-colors shadow-lg"
                  aria-label={t('upload.fileUpload.removeImage')}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-gray-700">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-gray-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Prompt Input */}
        <div className="space-y-4">
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700">
            {t('upload.prompt.label')}
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={t('upload.prompt.placeholder')}
            rows={5}
            className="w-full px-5 py-4 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900 placeholder-gray-500 bg-white resize-none text-base leading-relaxed"
            disabled={isLoading}
          />
          <p className="text-sm text-gray-500">
            {t('upload.prompt.description')}
          </p>
        </div>

        {/* Enhanced Start Verification Button */}
        <div className="flex justify-center pt-4">
          <Button
            type="submit"
            variant="primary"
            size="xl"
            disabled={!selectedFile || !prompt.trim() || isLoading}
            loading={isLoading}
            className="px-16 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-200"
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {t('upload.button.processing')}
              </span>
            ) : (
              t('upload.button.start')
            )}
          </Button>
        </div>
      </form>
    </div>
    </ErrorBoundary>
  )
}

export default UploadForm
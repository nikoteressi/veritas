/**
 * Custom hook for handling file upload operations.
 */
import { useState, useCallback } from 'react'
import { fileService } from '../services/fileService'
import { FileRejection } from 'react-dropzone'
import { useErrorHandler } from './useErrorHandler'
import { createFileError } from '../services/errorService'

interface UseFileUploadReturn {
  selectedFile: File | null
  previewUrl: string | null
  fileError: string | null
  handleFileSelection: (file: File | null) => boolean
  handleDropzoneFiles: (acceptedFiles: File[], rejectedFiles: FileRejection[]) => boolean
  clearFile: () => void
  cleanup: () => void
}

export const useFileUpload = (): UseFileUploadReturn => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const { handleError, clearError } = useErrorHandler()

  const handleFileSelection = useCallback((file: File | null): boolean => {
    setFileError(null)
    clearError()
    
    if (!file) {
      clearFile()
      return false
    }

    try {
      // Validate the file
      fileService.validateFile(file)
      
      // Set the file and create preview
      setSelectedFile(file)
      
      const url = fileService.createPreviewUrl(file)
      setPreviewUrl(url)
      
      return true
    } catch (error) {
      const fileError = createFileError(
        error instanceof Error ? error.message : 'File validation failed',
        file.name
      );
      handleError(fileError, { 
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type 
      });
      setFileError(fileError.message)
      return false
    }
  }, [clearError, handleError])

  const clearFile = useCallback(() => {
    setSelectedFile(null)
    setFileError(null)
    
    if (previewUrl) {
      fileService.revokePreviewUrl(previewUrl)
      setPreviewUrl(null)
    }
  }, [previewUrl])

  const handleDropzoneFiles = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]): boolean => {
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      if (rejection) {
        const rejectionError = createFileError(
          `File rejected: ${rejection.errors[0]?.message || 'Invalid file'}`,
          rejection.file.name
        );
        handleError(rejectionError, { 
          fileName: rejection.file.name,
          fileSize: rejection.file.size,
          fileType: rejection.file.type,
          errors: rejection.errors 
        });
        setFileError(rejectionError.message)
      }
      return false
    }

    const file = acceptedFiles[0]
    if (!file) {
      return false
    }
    return handleFileSelection(file)
  }, [handleFileSelection, handleError])

  // Cleanup preview URL on unmount
  const cleanup = useCallback(() => {
    if (previewUrl) {
      fileService.revokePreviewUrl(previewUrl)
    }
  }, [previewUrl])

  return {
    selectedFile,
    previewUrl,
    fileError,
    handleFileSelection,
    handleDropzoneFiles,
    clearFile,
    cleanup
  }
}
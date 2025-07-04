/**
 * Custom hook for handling file upload operations.
 */
import { useState, useCallback } from 'react';
import { fileService } from '../services/fileService';

export const useFileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileError, setFileError] = useState(null);

  const handleFileSelection = useCallback((file) => {
    setFileError(null);
    
    if (!file) {
      clearFile();
      return;
    }

    try {
      // Validate the file
      fileService.validateFile(file);
      
      // Set the file and create preview
      setSelectedFile(file);
      
      const url = fileService.createPreviewUrl(file);
      setPreviewUrl(url);
      
      return true;
    } catch (error) {
      setFileError(error.message);
      return false;
    }
  }, []);

  const clearFile = useCallback(() => {
    setSelectedFile(null);
    setFileError(null);
    
    if (previewUrl) {
      fileService.revokePreviewUrl(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  const handleDropzoneFiles = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      setFileError(`File rejected: ${rejection.errors[0]?.message || 'Invalid file'}`);
      return false;
    }

    const file = acceptedFiles[0];
    return handleFileSelection(file);
  }, [handleFileSelection]);

  // Cleanup preview URL on unmount
  const cleanup = useCallback(() => {
    if (previewUrl) {
      fileService.revokePreviewUrl(previewUrl);
    }
  }, [previewUrl]);

  return {
    selectedFile,
    previewUrl,
    fileError,
    handleFileSelection,
    handleDropzoneFiles,
    clearFile,
    cleanup
  };
}; 
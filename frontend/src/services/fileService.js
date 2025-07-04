/**
 * Service for handling file operations and validation.
 */
import { configurationService } from './configurationService';

export const fileService = {
  /**
   * Validate a file against size and type constraints.
   * @param {File} file - The file to validate
   * @throws {Error} If file validation fails
   */
  validateFile(file) {
    // Delegate to configuration service for validation
    return configurationService.validateFile(file);
  },

  /**
   * Create a preview URL for an image file.
   * @param {File} file - The image file
   * @returns {string} Object URL for preview
   */
  createPreviewUrl(file) {
    if (!file) return null;
    return URL.createObjectURL(file);
  },

  /**
   * Clean up a preview URL to prevent memory leaks.
   * @param {string} url - The object URL to revoke
   */
  revokePreviewUrl(url) {
    if (url) {
      URL.revokeObjectURL(url);
    }
  },

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
  },

  /**
   * Check if file type is supported.
   * @param {string} type - MIME type
   * @returns {boolean} Whether type is supported
   */
  isSupportedType(type) {
    return configurationService.isSupportedFileType(type);
  },

  /**
   * Get file extension from filename.
   * @param {string} filename - The filename
   * @returns {string} File extension
   */
  getFileExtension(filename) {
    if (!filename) return '';
    return filename.split('.').pop().toLowerCase();
  }
}; 
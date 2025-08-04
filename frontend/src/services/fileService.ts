/**
 * Service for handling file operations and validation.
 */
import { configurationService } from './configurationService'

interface FileService {
  validateFile(file: File): void
  createPreviewUrl(file: File | null): string | null
  revokePreviewUrl(url: string | null): void
  formatFileSize(bytes: number): string
  isSupportedType(type: string): boolean
  getFileExtension(filename: string): string
}

export const fileService: FileService = {
  /**
   * Validate a file against size and type constraints.
   */
  validateFile(file: File): boolean {
    configurationService.validateFile(file)
    return true
  },

  /**
   * Create a preview URL for an image file.
   */
  createPreviewUrl(file: File | null): string | null {
    if (!file) return null
    return URL.createObjectURL(file)
  },

  /**
   * Clean up a preview URL to prevent memory leaks.
   */
  revokePreviewUrl(url: string | null): void {
    if (url) {
      URL.revokeObjectURL(url)
    }
  },

  /**
   * Format file size for display.
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes'

    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  },

  /**
   * Check if file type is supported.
   */
  isSupportedType(type: string): boolean {
    return configurationService.isSupportedFileType(type)
  },

  /**
   * Get file extension from filename.
   */
  getFileExtension(filename: string): string {
    if (!filename) return ''
    return filename.split('.').pop()?.toLowerCase() || ''
  }
}
import React from 'react';

interface ScreenshotDisplayProps {
  imageUrl?: string;
  filename?: string;
  fileSize?: number;
  className?: string;
  showFileInfo?: boolean;
  maxHeight?: string;
}

/**
 * Component for displaying uploaded screenshots in verification results
 * Features:
 * - Responsive image display
 * - File information overlay
 * - Proper aspect ratio handling
 * - Loading states
 */
const ScreenshotDisplay: React.FC<ScreenshotDisplayProps> = React.memo(function ScreenshotDisplay({ 
  imageUrl, 
  filename, 
  fileSize, 
  className = '',
  showFileInfo = true,
  maxHeight = '400px'
}) {
  if (!imageUrl) {
    return null;
  }

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'Unknown size';
    return (bytes / 1024 / 1024).toFixed(2) + ' MB';
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>): void => {
    const target = e.target as HTMLImageElement;
    target.style.display = 'none';
    const nextSibling = target.nextSibling as HTMLElement;
    if (nextSibling) {
      nextSibling.style.display = 'flex';
    }
  };

  return (
    <div className={`bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm ${className}`}>
      {/* Image Container */}
      <div className="relative">
        <img
          src={imageUrl}
          alt={filename || 'Uploaded screenshot'}
          className="w-full object-contain bg-gray-50"
          style={{ maxHeight }}
          onError={handleImageError}
        />
        
        {/* Error fallback */}
        <div 
          className="hidden w-full h-64 bg-gray-100 items-center justify-center text-gray-500"
          style={{ minHeight: '200px' }}
        >
          <div className="text-center">
            <svg className="w-12 h-12 mx-auto mb-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm">Image could not be loaded</p>
          </div>
        </div>
      </div>

      {/* File Information */}
      {showFileInfo && (filename || fileSize) && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
              </svg>
              <span className="text-gray-700 font-medium truncate max-w-xs">
                {filename || 'Uploaded image'}
              </span>
            </div>
            {fileSize && (
              <span className="text-gray-500 ml-2">
                {formatFileSize(fileSize)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default ScreenshotDisplay;
import React from 'react'
import ReputationDisplay from '../ReputationDisplay'
import ScreenshotDisplay from '../ui/ScreenshotDisplay'
import ResultCard from '../ui/ResultCard'

function ResultDisplay({ result }) {
  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'true':
        return 'text-green-700 bg-green-50 border-green-200'
      case 'partially_true':
        return 'text-yellow-700 bg-yellow-50 border-yellow-200'
      case 'false':
        return 'text-red-700 bg-red-50 border-red-200'
      case 'ironic':
        return 'text-blue-700 bg-blue-50 border-blue-200'
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200'
    }
  }

  const formatVerdict = (verdict) => {
    switch (verdict) {
      case 'true':
        return 'True'
      case 'partially_true':
        return 'Partially True'
      case 'false':
        return 'False'
      case 'ironic':
        return 'Ironic/Satirical'
      default:
        return verdict
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        )
      case 'error':
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        )
      default:
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        )
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        Verification Results
      </h2>

      {/* Enhanced Status Banner */}
      <div className={`rounded-xl border-2 p-6 mb-8 ${getStatusColor(result.status)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              {getStatusIcon(result.status)}
            </div>
            <div className="ml-3">
              <h3 className="text-lg font-semibold">
                {result.status === 'success' ? 'Verification Complete' : 
                 result.status === 'error' ? 'Verification Failed' : 'Processing'}
              </h3>
              {result.message && (
                <p className="mt-1 text-sm opacity-90">{result.message}</p>
              )}
            </div>
          </div>
          {result.status === 'success' && (
            <div className="text-right">
              <div className="text-sm opacity-75">Analysis completed</div>
              {result.processing_time && (
                <div className="text-xs opacity-60">
                  Processed in {result.processing_time}s
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Results Details */}
      {result.status === 'success' && (
        <div className="space-y-6">
          {/* Screenshot Display */}
          {(result.image_url || result.screenshot_url || result.file_url) && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Analyzed Image</h3>
              <ScreenshotDisplay
                imageUrl={result.image_url || result.screenshot_url || result.file_url}
                filename={result.filename}
                fileSize={result.file_size}
                maxHeight="500px"
              />
            </div>
          )}

          {/* File Information Section */}
          <ResultCard 
            title="File Information" 
            colorScheme="gray"
            className="mt-0"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {result.filename && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">Filename</h4>
                  <p className="text-gray-600 text-sm break-all">{result.filename}</p>
                </div>
              )}

              {result.file_size && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">File Size</h4>
                  <p className="text-gray-600 text-sm">{(result.file_size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              )}

              {result.processing_time && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">Processing Time</h4>
                  <p className="text-gray-600 text-sm">{result.processing_time} seconds</p>
                </div>
              )}
            </div>

            {result.prompt && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h4 className="font-medium text-gray-700 mb-2">Your Question</h4>
                <p className="text-gray-600 text-sm bg-white rounded-lg p-3 border">{result.prompt}</p>
              </div>
            )}
          </ResultCard>

          {/* Verdict Display */}
          {result.verdict && (
            <div className={`rounded-xl border-2 p-6 ${getVerdictColor(result.verdict)}`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-3">
                    Verdict: {formatVerdict(result.verdict)}
                  </h3>
                  {result.confidence_score && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="font-medium">Confidence Level</span>
                        <span className="font-semibold">{result.confidence_score}%</span>
                      </div>
                      <div className="w-full bg-white bg-opacity-50 rounded-full h-2">
                        <div 
                          className="bg-current h-2 rounded-full transition-all duration-300"
                          style={{ width: `${result.confidence_score}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  {result.justification && (
                    <div>
                      <h4 className="font-medium mb-2">Analysis</h4>
                      <p className="text-sm leading-relaxed">{result.justification}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Claims and Analysis */}
          {result.identified_claims && result.identified_claims.length > 0 && (
            <ResultCard 
              title="Identified Claims" 
              colorScheme="blue"
              icon={
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
              }
            >
              <div className="space-y-3">
                {result.identified_claims.map((claim, index) => (
                  <div key={index} className="bg-white rounded-lg p-4 border border-blue-100">
                    <div className="flex items-start">
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium mr-3 mt-0.5">
                        {index + 1}
                      </span>
                      <p className="text-gray-700 text-sm leading-relaxed">{claim}</p>
                    </div>
                  </div>
                ))}
              </div>
            </ResultCard>
          )}

          {/* Sources */}
          {result.sources && result.sources.length > 0 && (
            <ResultCard 
              title={`Sources Consulted (${result.sources.length})`} 
              colorScheme="green"
              icon={
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z" clipRule="evenodd" />
                </svg>
              }
            >
              <div className="space-y-3">
                {result.sources.map((source, index) => (
                  <div key={index} className="bg-white rounded-lg p-4 border border-green-100">
                    <div className="flex items-start">
                      <span className="flex-shrink-0 w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-medium mr-3 mt-0.5">
                        {index + 1}
                      </span>
                      <a 
                        href={source} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-green-700 hover:text-green-900 underline break-all text-sm leading-relaxed flex-1"
                      >
                        {source}
                      </a>
                      <svg className="w-4 h-4 text-green-600 ml-2 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </div>
                  </div>
                ))}
              </div>
            </ResultCard>
          )}

          {/* Motives Analysis */}
          {result.motives_analysis && (
            <ResultCard 
              title="Motives Analysis" 
              colorScheme="orange"
              icon={
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z" clipRule="evenodd" />
                </svg>
              }
            >
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Primary Analysis */}
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border border-orange-100">
                    <h4 className="font-semibold text-orange-800 mb-2">Primary Motive</h4>
                    <p className="text-gray-700 capitalize text-sm">
                      {result.motives_analysis.primary_motive?.replace(/_/g, ' ')}
                    </p>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 border border-orange-100">
                    <h4 className="font-semibold text-orange-800 mb-2">Confidence Score</h4>
                    <div className="flex items-center">
                      <div className="flex-1 mr-3">
                        <div className="w-full bg-orange-200 rounded-full h-2">
                          <div 
                            className="bg-orange-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${Math.round(result.motives_analysis.confidence_score * 100)}%` }}
                          ></div>
                        </div>
                      </div>
                      <span className="text-sm font-medium text-orange-800">
                        {Math.round(result.motives_analysis.confidence_score * 100)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Risk Assessment */}
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border border-orange-100">
                    <h4 className="font-semibold text-orange-800 mb-2">Credibility Assessment</h4>
                    <p className="text-gray-700 capitalize text-sm">
                      {result.motives_analysis.credibility_assessment?.replace(/_/g, ' ')}
                    </p>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 border border-orange-100">
                    <h4 className="font-semibold text-orange-800 mb-2">Risk Level</h4>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium capitalize ${
                      result.motives_analysis.risk_level === 'high' ? 'bg-red-100 text-red-800' :
                      result.motives_analysis.risk_level === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {result.motives_analysis.risk_level}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Manipulation Indicators */}
              {result.motives_analysis.manipulation_indicators && result.motives_analysis.manipulation_indicators.length > 0 && (
                <div className="mt-6 pt-6 border-t border-orange-200">
                  <h4 className="font-semibold text-orange-800 mb-4 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    Manipulation Indicators
                  </h4>
                  <div className="grid grid-cols-1 gap-3">
                    {result.motives_analysis.manipulation_indicators.slice(0, 5).map((indicator, index) => (
                      <div key={index} className="bg-white rounded-lg p-3 border border-orange-100">
                        <div className="flex items-start">
                          <span className="flex-shrink-0 w-5 h-5 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs font-medium mr-3 mt-0.5">
                            !
                          </span>
                          <p className="text-gray-700 text-sm leading-relaxed">{indicator}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </ResultCard>
          )}

          {/* Processing Info */}
          <ResultCard 
            title="Processing Details" 
            colorScheme="gray"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {result.processing_time_seconds && (
                <div className="flex items-center">
                  <svg className="w-4 h-4 text-gray-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  <span className="text-gray-600 text-sm">Processing Time:</span>
                  <span className="ml-2 font-medium text-sm">{result.processing_time_seconds}s</span>
                </div>
              )}
              {result.primary_topic && (
                <div className="flex items-center">
                  <svg className="w-4 h-4 text-gray-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M17.707 9.293a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-7-7A.997.997 0 012 10V5a3 3 0 013-3h5c.256 0 .512.098.707.293l7 7zM5 6a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                  </svg>
                  <span className="text-gray-600 text-sm">Topic:</span>
                  <span className="ml-2 font-medium text-sm capitalize">{result.primary_topic}</span>
                </div>
              )}
            </div>
          </ResultCard>
        </div>
      )}

      {/* User Reputation Display */}
      {result.status === 'success' && result.user_reputation && (
        <ResultCard 
          title="User Reputation" 
          colorScheme="purple"
          icon={
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
          }
        >
          <ReputationDisplay
            reputation={result.user_reputation}
            warnings={result.warnings || []}
          />
        </ResultCard>
      )}
    </div>
  )
}

export default ResultDisplay
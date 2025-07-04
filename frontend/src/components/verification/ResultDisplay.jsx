import React from 'react'
import ReputationDisplay from '../ReputationDisplay'

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
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Verification Results
      </h2>

      {/* Status Banner */}
      <div className={`rounded-lg border p-4 mb-6 ${getStatusColor(result.status)}`}>
        <div className="flex items-center">
          {getStatusIcon(result.status)}
          <span className="ml-2 font-medium">
            {result.status === 'success' ? 'Verification Complete' : 
             result.status === 'error' ? 'Verification Failed' : 'Processing'}
          </span>
        </div>
        {result.message && (
          <p className="mt-2 text-sm">{result.message}</p>
        )}
      </div>

      {/* Results Details */}
      {result.status === 'success' && (
        <div className="space-y-4">
          {result.filename && (
            <div>
              <h3 className="font-medium text-gray-800 mb-1">File Processed</h3>
              <p className="text-gray-600">{result.filename}</p>
            </div>
          )}

          {result.prompt && (
            <div>
              <h3 className="font-medium text-gray-800 mb-1">Your Question</h3>
              <p className="text-gray-600">{result.prompt}</p>
            </div>
          )}

          {result.file_size && (
            <div>
              <h3 className="font-medium text-gray-800 mb-1">File Size</h3>
              <p className="text-gray-600">{(result.file_size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          )}

          {/* Verdict Display */}
          {result.verdict && (
            <div className={`rounded-lg border p-4 mb-4 ${getVerdictColor(result.verdict)}`}>
              <h3 className="font-medium mb-2">Verdict: {formatVerdict(result.verdict)}</h3>
              {result.confidence_score && (
                <p className="text-sm mb-2">Confidence: {result.confidence_score}%</p>
              )}
              {result.justification && (
                <p className="text-sm">{result.justification}</p>
              )}
            </div>
          )}

          {/* Claims and Analysis */}
          {result.identified_claims && result.identified_claims.length > 0 && (
            <div className="mb-4">
              <h3 className="font-medium text-gray-800 mb-2">Identified Claims</h3>
              <ul className="space-y-1">
                {result.identified_claims.map((claim, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-start">
                    <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                    {claim}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Sources */}
          {result.sources && result.sources.length > 0 && (
            <div className="mb-4">
              <h3 className="font-medium text-gray-800 mb-2">Sources Consulted</h3>
              <div className="bg-gray-50 rounded-lg p-3">
                <ul className="space-y-2">
                  {result.sources.map((source, index) => (
                    <li key={index} className="text-sm">
                      <a 
                        href={source} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 underline break-all"
                      >
                        {source}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Motives Analysis */}
          {result.motives_analysis && (
            <div className="mb-4">
              <h3 className="font-medium text-gray-800 mb-2">Motives Analysis</h3>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Primary Motive:</span>
                    <span className="ml-2 capitalize">{result.motives_analysis.primary_motive?.replace(/_/g, ' ')}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Confidence:</span>
                    <span className="ml-2">{Math.round(result.motives_analysis.confidence_score * 100)}%</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Credibility:</span>
                    <span className="ml-2 capitalize">{result.motives_analysis.credibility_assessment?.replace(/_/g, ' ')}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Risk Level:</span>
                    <span className={`ml-2 capitalize ${
                      result.motives_analysis.risk_level === 'high' ? 'text-red-600' :
                      result.motives_analysis.risk_level === 'moderate' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {result.motives_analysis.risk_level}
                    </span>
                  </div>
                </div>
                
                {result.motives_analysis.manipulation_indicators && result.motives_analysis.manipulation_indicators.length > 0 && (
                  <div className="mt-3">
                    <h4 className="font-medium text-gray-700 mb-2">Manipulation Indicators:</h4>
                    <ul className="space-y-1">
                      {result.motives_analysis.manipulation_indicators.slice(0, 5).map((indicator, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                          {indicator}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Processing Info */}
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <h3 className="font-medium text-gray-800 mb-2">Processing Details</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              {result.processing_time_seconds && (
                <div>
                  <span className="text-gray-600">Processing Time:</span>
                  <span className="ml-1 font-medium">{result.processing_time_seconds}s</span>
                </div>
              )}
              {result.primary_topic && (
                <div>
                  <span className="text-gray-600">Topic:</span>
                  <span className="ml-1 font-medium capitalize">{result.primary_topic}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* User Reputation Display */}
      {result.status === 'success' && result.user_reputation && (
        <div className="mt-6">
          <ReputationDisplay
            reputation={result.user_reputation}
            warnings={result.warnings || []}
          />
        </div>
      )}
    </div>
  )
}

export default ResultDisplay 
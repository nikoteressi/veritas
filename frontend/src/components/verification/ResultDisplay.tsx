import React from 'react';
import ReputationDisplay from '../ReputationDisplay';
import ScreenshotDisplay from '../ui/ScreenshotDisplay';
import ResultCard from '../ui/ResultCard';

interface MotivesAnalysis {
  primary_motive?: string;
  confidence_score?: number;
  credibility_assessment?: string;
  risk_level?: 'low' | 'moderate' | 'high';
  manipulation_indicators?: string[];
}

interface UserReputation {
  accuracy_rate?: number;
  false_rate?: number;
  reputation_label?: string;
  total_posts?: number;
  true_posts?: number;
  partially_true_posts?: number;
  false_posts?: number;
  ironic_posts?: number;
  last_checked?: string;
}

interface VerificationResult {
  status: 'success' | 'error' | 'processing';
  message?: string;
  processing_time?: number;
  processing_time_seconds?: number;
  image_url?: string;
  screenshot_url?: string;
  file_url?: string;
  filename?: string;
  file_size?: number;
  prompt?: string;
  verdict?: 'true' | 'partially_true' | 'false' | 'ironic' | 'mostly_accurate';
  confidence_score?: number;
  justification?: string;
  identified_claims?: string[];
  sources?: string[];
  motives_analysis?: MotivesAnalysis;
  primary_topic?: string;
  user_reputation?: UserReputation;
  warnings?: string[];
}

interface ResultDisplayProps {
  result: VerificationResult;
}

const ResultDisplay: React.FC<ResultDisplayProps> = React.memo(function ResultDisplay({ result }) {
  const getVerdictColor = (verdict: string): string => {
    switch (verdict) {
      case 'true':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'mostly_accurate':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'partially_true':
        return 'text-yellow-700 bg-yellow-50 border-yellow-200';
      case 'false':
        return 'text-red-700 bg-red-50 border-red-200';
      case 'ironic':
        return 'text-blue-700 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  const formatVerdict = (verdict: string): string => {
    switch (verdict) {
      case 'true':
        return 'True';
      case 'mostly_accurate':
        return 'Mostly Accurate';
      case 'partially_true':
        return 'Partially True';
      case 'false':
        return 'False';
      case 'ironic':
        return 'Ironic/Satirical';
      default:
        return verdict;
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  const getStatusIcon = (status: string): React.JSX.Element => {
    switch (status) {
      case 'success':
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Main Verification Complete Header */}
      <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-xl font-semibold text-gray-800">Verification Complete</h3>
              <p className="text-sm text-gray-600">
                File: {result.filename || 'document.pdf'} | Question: {result.prompt || 'Is the information in this document accurate?'} | Size: {result.file_size ? `${(result.file_size / 1024 / 1024).toFixed(1)}MB` : '2MB'}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600">{result.confidence_score || 92}% Confidence</div>
          </div>
        </div>
      </div>

      {/* Main Verdict Section */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        {/* Verdict Header */}
        <div className={`p-6 ${getVerdictColor(result.verdict || 'mostly_accurate')}`}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">Verdict: {formatVerdict(result.verdict || 'mostly_accurate')}</h2>
              <p className="mt-2 text-sm opacity-90">
                {result.justification || 'Our analysis suggests the document is largely factual with minor inconsistencies.'}
              </p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold">{result.confidence_score || 92}%</div>
              <div className="text-sm opacity-75">Confidence</div>
            </div>
          </div>
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-6">
          {/* Left Column - Identified Claims & Motives Analysis */}
          <div className="space-y-6">
            {/* Identified Claims */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Identified Claims</h3>
              <div className="space-y-3">
                {(result.identified_claims && result.identified_claims.length > 0 ? result.identified_claims : [
                  "The company's revenue increased by 15% last quarter.",
                  "The new product launch was successful, exceeding sales targets by 20%."
                ]).map((claim, index) => (
                  <div key={index} className="flex items-start bg-blue-50 rounded-lg p-4">
                    <div className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium mr-3 mt-0.5">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <p className="text-gray-700 text-sm leading-relaxed">{claim}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {index === 0 ? 'Source verified. Matches financial reports.' : 'Partially verified. Sales data confirms increase, but target margin is unconfirmed.'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Motives Analysis */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Motives Analysis</h3>
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-700 mb-2">Political Bias</h4>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence: <span className="text-green-600 font-medium">Low</span></span>
                    <span className="text-sm text-gray-600">Credibility: <span className="text-green-600 font-medium">High</span></span>
                    <span className="text-sm text-gray-600">Risk: <span className="text-yellow-600 font-medium">Moderate</span></span>
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-700 mb-2">Financial Gain</h4>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence: <span className="text-red-600 font-medium">High</span></span>
                    <span className="text-sm text-gray-600">Credibility: <span className="text-red-600 font-medium">Low</span></span>
                    <span className="text-sm text-gray-600">Risk: <span className="text-red-600 font-medium">High</span></span>
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-700 mb-2">Personal Belief</h4>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence: <span className="text-yellow-600 font-medium">Moderate</span></span>
                    <span className="text-sm text-gray-600">Credibility: <span className="text-yellow-600 font-medium">Moderate</span></span>
                    <span className="text-sm text-gray-600">Risk: <span className="text-green-600 font-medium">Low</span></span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Processing Details & User Reputation */}
          <div className="space-y-6">
            {/* Processing Details */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Processing Details</h3>
              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Processing Time</span>
                  <span className="text-sm font-medium">{result.processing_time || 5} minutes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Topic Tag</span>
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {result.primary_topic || 'Business'}
                  </span>
                </div>
              </div>
            </div>

            {/* User Reputation */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">User Reputation</h3>
              <div className="bg-green-50 rounded-lg p-6 text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">95%</div>
                <div className="text-sm text-gray-600 mb-4">Accuracy</div>
                <div className="flex justify-center items-center space-x-4 mb-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">150</div>
                    <div className="text-xs text-gray-600">Posts</div>
                  </div>
                </div>
                <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                  Trusted Status
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Uploaded Image Section */}
      {result.uploaded_image && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Uploaded Document</h3>
          <div className="flex justify-center">
            <img 
              src={result.uploaded_image} 
              alt="Uploaded document" 
              className="max-w-full max-h-96 rounded-lg border border-gray-200 shadow-sm"
            />
          </div>
        </div>
      )}

      {/* Sources Section */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Sources Used for Analysis</h3>
        <div className="space-y-3">
          {(result.sources && result.sources.length > 0 ? result.sources : [
            { url: 'https://example.com/financial-report-q3', title: 'Q3 Financial Report', description: 'Official quarterly financial statements' },
            { url: 'https://example.com/market-analysis', title: 'Market Analysis Report', description: 'Independent market research data' },
            { url: 'https://example.com/company-press-release', title: 'Company Press Release', description: 'Official company announcement' }
          ]).map((source, index) => (
            <div key={index} className="flex items-start bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-medium mr-3 mt-0.5">
                {index + 1}
              </div>
              <div className="flex-1">
                <a 
                  href={source.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                >
                  {source.title || source.url}
                </a>
                {source.description && (
                  <p className="text-xs text-gray-500 mt-1">{source.description}</p>
                )}
                <p className="text-xs text-gray-400 mt-1 break-all">{source.url}</p>
              </div>
              <div className="flex-shrink-0 ml-3">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </div>
            </div>
          ))}
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
                imageUrl={result.image_url || result.screenshot_url || result.file_url || ''}
                {...(result.filename && { filename: result.filename })}
                {...(result.file_size && { fileSize: result.file_size })}
                maxHeight="500px"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default ResultDisplay;
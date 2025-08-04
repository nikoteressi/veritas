import React from 'react';
import { useTranslation } from 'react-i18next';
import ErrorDisplay from '../ui/ErrorDisplay';
import { AppError, ErrorSeverity } from '../../types';

interface ErrorStateProps {
  message?: string;
  error?: AppError;
}

const ErrorState: React.FC<ErrorStateProps> = ({ message, error }) => {
  const { t } = useTranslation();

  // Create AppError from message if needed
  const displayError = error || (message ? {
    message,
    code: 'VERIFICATION_ERROR',
    severity: 'high' as ErrorSeverity,
    timestamp: new Date().toISOString(),
    details: {}
  } : {
    message: t('verification.error.title'),
    code: 'VERIFICATION_ERROR',
    severity: 'high' as ErrorSeverity,
    timestamp: new Date().toISOString(),
    details: {}
  });

  return (
    <div className="text-center py-8">
      <ErrorDisplay 
        error={displayError}
        size="lg"
        showDetails={true}
      />
    </div>
  );
};

export default ErrorState;
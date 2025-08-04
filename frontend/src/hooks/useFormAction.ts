import { useState, useCallback } from 'react';

interface FormState<T = any> {
  data?: T;
  error?: string;
  success?: boolean;
  pending?: boolean;
}

interface UseFormActionOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: string) => void;
  resetOnSuccess?: boolean;
}

/**
 * Modern React 19 hook for handling form actions with built-in state management
 * Uses useActionState and useFormStatus for optimal UX
 */
export function useFormAction<T = any>(
  action: (prevState: FormState<T>, formData: FormData) => Promise<FormState<T>>,
  initialState: FormState<T> = {},
  options: UseFormActionOptions<T> = {}
) {
  const { onSuccess, onError, resetOnSuccess = true } = options;
  const [state, setState] = useState<FormState<T>>(initialState);

  const formAction = useCallback(async (formData: FormData) => {
    setState(prev => ({ ...prev, pending: true }));
    
    try {
      const currentState = state;
      const result = await action(currentState, formData);
      
      if (result.success && onSuccess && result.data) {
        onSuccess(result.data);
      }
      
      if (result.error && onError) {
        onError(result.error);
      }
      
      const newState = resetOnSuccess && result.success 
        ? { ...initialState, pending: false }
        : { ...result, pending: false };
        
      setState(newState);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      if (onError) {
        onError(errorMessage);
      }
      setState({ error: errorMessage, pending: false });
    }
  }, [action, onSuccess, onError, resetOnSuccess, initialState]);

  return {
    state,
    formAction,
    isLoading: state.pending || false,
    error: state.error,
    success: state.success,
    data: state.data
  };
}

/**
 * Hook to get form submission status
 */
export function useFormSubmissionStatus() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  return {
    isSubmitting,
    setIsSubmitting,
    submissionData: null,
    submissionMethod: null,
    submissionAction: null
  };
}
// Core verification types
export interface VerificationResult {
  id?: string;
  status: 'pending' | 'completed' | 'failed';
  confidence?: number;
  analysis?: string;
  sources?: Source[];
  timestamp: string;
  message?: string;
  metadata?: {
    processingTime?: number;
    modelVersion?: string;
    [key: string]: any;
  };
}

export interface Source {
  url: string;
  title: string;
  credibility: number;
  relevance: number;
  snippet?: string;
  publishDate?: string;
  domain?: string;
}

// WebSocket types
export type WebSocketMessageType = 
  | 'steps_definition'
  | 'progress_update'
  | 'step_update'
  | 'session_established'
  | 'verification_complete'
  | 'start_verification'
  | 'error'
  | 'ping'
  | 'pong';

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  data: T;
  timestamp: number;
  sessionId?: string;
}

// Specific WebSocket message data types
export interface StepsDefinitionData {
  steps: ProgressStep[];
}

export interface ProgressUpdateData {
  current_progress: number;
  message?: string;
  animation_duration?: number;
}

export interface StepUpdateData {
  step_id: string;
  status: StepStatus;
  progress?: number;
  message?: string;
  metadata?: Record<string, unknown>;
}

export interface SessionEstablishedData {
  session_id: string;
  timestamp: string;
}

export interface VerificationCompleteData {
  result: VerificationResult;
  session_id: string;
}

export interface ErrorData {
  message: string;
  code?: string;
  details?: Record<string, unknown>;
}

export interface WebSocketState {
  isConnected: boolean;
  sessionId: string | null;
  reconnectAttempts: number;
  lastMessage?: WebSocketMessage | undefined;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  wsError?: string;
}

// Verification state types
export interface VerificationState {
  verificationResult: VerificationResult | null;
  isLoading: boolean;
  error: string | null;
  progress?: ProgressState;
}

// Progress and Step types (unified)
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface ProgressStep {
  id: string;
  title: string;
  description?: string;
  status: StepStatus;
  startTime?: string;
  endTime?: string;
  duration?: number;
  progress?: number; // 0-100
  substeps?: ProgressSubstep[];
  metadata?: Record<string, any>;
}

export interface ProgressSubstep {
  id: string;
  title: string;
  status: StepStatus;
  progress?: number;
}

export interface ProgressState {
  steps: ProgressStep[];
  currentStepIndex: number;
  overallProgress: number; // 0-100
  isActive: boolean;
  startTime?: string;
  estimatedTimeRemaining?: number;
  totalDuration?: number;
}

// File upload types
export interface FileUploadState {
  file: File | null;
  preview: string | null;
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
}

// API response types
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface VerificationRequest {
  file: File;
  prompt: string;
  sessionId: string;
  options?: {
    priority?: 'low' | 'normal' | 'high';
    timeout?: number;
    [key: string]: any;
  };
}

// Component prop types
export interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
}

// Context types
export interface VerificationContextType extends VerificationState {
  startVerification: () => void;
  completeVerification: (result: VerificationResult) => void;
  setError: (error: string | null) => void;
  clearState: () => void;
  reset: () => void;
  handleWebSocketMessage: <T = unknown>(type: WebSocketMessageType, data: T) => void;
  getVerificationSummary: () => string;
}

export interface WebSocketContextType extends WebSocketState {
  sendMessage: <T = unknown>(type: WebSocketMessageType, data?: T) => void;
  subscribeToMessage: <T = unknown>(type: WebSocketMessageType, callback: (data: T) => void) => () => void;
  reconnect: () => Promise<void>;
  getConnectionInfo: () => WebSocketState;
}

// WebSocket Service interface
export interface WebSocketService {
  connect: (url?: string | null) => Promise<string>;
  disconnect: () => void;
  send: (type: string, data?: any) => boolean;
  subscribe: (eventType: string, callback: (data?: any) => void) => () => void;
  getStatus: () => {
    isConnected: boolean;
    sessionId: string | null;
    reconnectAttempts: number;
    readyState: number;
  };
}

// Hook return types
export interface UseVeritasReturn extends VerificationState, WebSocketState {
  submitVerification: (file: File, prompt: string, sessionId: string, isConnected: boolean) => Promise<void>;
  onVerificationStart: () => void;
  onVerificationComplete: (result: VerificationResult) => void;
  clearState: () => void;
  reset: () => void;
  sendMessage: <T = unknown>(type: WebSocketMessageType, data?: T) => void;
  subscribeToMessage: <T = unknown>(type: WebSocketMessageType, callback: (data: T) => void) => () => void;
  reconnect: () => Promise<void>;
  getVerificationSummary: () => VerificationSummary;
  getConnectionInfo: () => WebSocketState;
}

export interface VerificationSummary {
  hasResult: boolean;
  isProcessing: boolean;
  hasError: boolean;
  status: 'idle' | 'loading' | 'completed' | 'error';
}

export interface UseFileUploadReturn extends FileUploadState {
  handleFileSelect: (file: File) => void;
  handleFileRemove: () => void;
  uploadFile: (file: File) => Promise<string>;
  resetUpload: () => void;
}

export interface UseWebSocketServiceReturn {
  isConnected: boolean;
  sessionId: string | null;
  reconnectAttempts: number;
  wsError: string | null;
  lastMessage: WebSocketMessage | null;
  sendMessage: <T = unknown>(type: WebSocketMessageType, data?: T) => void;
  subscribeToMessage: <T = unknown>(messageType: WebSocketMessageType, callback: (data: T) => void) => () => void;
  reconnect: () => Promise<void>;
  getConnectionInfo: () => WebSocketState;
  webSocketService: WebSocketService;
}

// Utility types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export type Theme = 'light' | 'dark' | 'system';

export interface AppConfig {
  apiBaseUrl: string;
  wsBaseUrl: string;
  maxFileSize: number;
  supportedFileTypes: string[];
  theme: Theme;
  language: string;
}

// Error types
export interface AppError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
  stack?: string;
  severity: ErrorSeverity;
  cause?: Error;
}

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface ErrorInfo {
  error: AppError;
  severity: ErrorSeverity;
  context?: Record<string, any>;
}

// API response types
export interface ReputationData {
  nickname: string;
  score: number;
  level: string;
  badges: string[];
  verificationCount: number;
  accuracy: number;
  lastActivity: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  version: string;
  timestamp: string;
  services: {
    database: 'up' | 'down';
    redis: 'up' | 'down';
    websocket: 'up' | 'down';
  };
}

// File rejection types for react-dropzone
export interface FileRejection {
  file: File
  errors: Array<{
    code: string
    message: string
  }>
}

// Vite environment variables
declare global {
  interface ImportMetaEnv {
    readonly VITE_API_BASE_URL: string
    readonly VITE_API_TIMEOUT: string
    readonly VITE_MAX_FILE_SIZE: string
    readonly VITE_WS_URL: string
    readonly VITE_CONNECTION_TIMEOUT: string
    readonly VITE_APP_VERSION: string
    readonly MODE: string
    readonly DEV: boolean
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv
  }
}
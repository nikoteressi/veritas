import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from '../../App'

// Mock the contexts
vi.mock('../../contexts/VerificationContext', () => ({
  VerificationProvider: ({ children }) => children,
  useVerificationContext: () => ({
    verificationResult: null,
    isLoading: false,
    error: null,
    onVerificationStart: vi.fn(),
    onVerificationComplete: vi.fn()
  })
}))

vi.mock('../../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }) => children,
  useWebSocketContext: () => ({
    isConnected: false,
    sessionId: null
  })
}))

// Mock react-i18next
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key) => key,
    i18n: {
      changeLanguage: () => new Promise(() => {})
    }
  })
}))

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />)
    expect(screen.getByRole('banner')).toBeInTheDocument()
  })

  it('renders skip to content link', () => {
    render(<App />)
    expect(screen.getByText('accessibility.skipToContent')).toBeInTheDocument()
  })
})
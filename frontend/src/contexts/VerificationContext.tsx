import React, { createContext, useContext, ReactNode } from 'react'
import { useVeritas } from '../hooks/useVeritas'
import { UseVeritasReturn } from '../types'

interface VerificationProviderProps {
  children: ReactNode
}

const VerificationContext = createContext<UseVeritasReturn | null>(null)

export const useVerificationContext = (): UseVeritasReturn => {
  const context = useContext(VerificationContext)
  if (!context) {
    throw new Error('useVerificationContext must be used within a VerificationProvider')
  }
  return context
}

export const VerificationProvider: React.FC<VerificationProviderProps> = ({ children }) => {
  const veritasState = useVeritas()

  return (
    <VerificationContext.Provider value={veritasState}>
      {children}
    </VerificationContext.Provider>
  )
}
import React from 'react'

const Card = ({ 
  children, 
  className = '', 
  variant = 'default',
  padding = 'default',
  shadow = 'default',
  ...props 
}) => {
  const baseClasses = 'rounded-lg border'
  
  const variantClasses = {
    default: 'bg-white border-gray-200',
    success: 'bg-green-50 border-green-200',
    warning: 'bg-yellow-50 border-yellow-200',
    danger: 'bg-red-50 border-red-200',
    info: 'bg-blue-50 border-blue-200'
  }
  
  const paddingClasses = {
    none: '',
    sm: 'p-3',
    default: 'p-4',
    lg: 'p-6',
    xl: 'p-8'
  }
  
  const shadowClasses = {
    none: '',
    sm: 'shadow-sm',
    default: 'shadow-md',
    lg: 'shadow-lg',
    xl: 'shadow-xl'
  }
  
  const classes = [
    baseClasses,
    variantClasses[variant],
    paddingClasses[padding],
    shadowClasses[shadow],
    className
  ].join(' ')
  
  return (
    <div className={classes} {...props}>
      {children}
    </div>
  )
}

const CardHeader = ({ children, className = '', ...props }) => {
  return (
    <div className={`border-b border-gray-200 pb-3 mb-4 ${className}`} {...props}>
      {children}
    </div>
  )
}

const CardTitle = ({ children, className = '', ...props }) => {
  return (
    <h3 className={`text-lg font-semibold text-gray-900 ${className}`} {...props}>
      {children}
    </h3>
  )
}

const CardContent = ({ children, className = '', ...props }) => {
  return (
    <div className={className} {...props}>
      {children}
    </div>
  )
}

const CardFooter = ({ children, className = '', ...props }) => {
  return (
    <div className={`border-t border-gray-200 pt-3 mt-4 ${className}`} {...props}>
      {children}
    </div>
  )
}

// Export individual components
Card.Header = CardHeader
Card.Title = CardTitle
Card.Content = CardContent
Card.Footer = CardFooter

export default Card
export { CardHeader, CardTitle, CardContent, CardFooter }
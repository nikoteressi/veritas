import React from 'react';

type ColorScheme = 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'gray' | 'indigo' | 'orange';

interface ResultCardProps {
  title?: string;
  icon?: React.ReactNode;
  colorScheme?: ColorScheme;
  children: React.ReactNode;
  className?: string;
}

interface ColorSchemeConfig {
  bg: string;
  border: string;
  titleColor: string;
  iconColor: string;
}

/**
 * Переиспользуемый компонент карточки для отображения результатов верификации
 */
const ResultCard: React.FC<ResultCardProps> = React.memo(({ 
  title, 
  icon, 
  colorScheme = 'gray', 
  children, 
  className = '' 
}) => {
  // Определяем цветовые схемы
  const colorSchemes: Record<ColorScheme, ColorSchemeConfig> = {
    blue: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      titleColor: 'text-blue-900',
      iconColor: 'text-blue-600'
    },
    green: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      titleColor: 'text-green-900',
      iconColor: 'text-green-600'
    },
    red: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      titleColor: 'text-red-900',
      iconColor: 'text-red-600'
    },
    yellow: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      titleColor: 'text-yellow-900',
      iconColor: 'text-yellow-600'
    },
    purple: {
      bg: 'bg-purple-50',
      border: 'border-purple-200',
      titleColor: 'text-purple-900',
      iconColor: 'text-purple-600'
    },
    gray: {
      bg: 'bg-gray-50',
      border: 'border-gray-200',
      titleColor: 'text-gray-900',
      iconColor: 'text-gray-600'
    },
    indigo: {
      bg: 'bg-indigo-50',
      border: 'border-indigo-200',
      titleColor: 'text-indigo-900',
      iconColor: 'text-indigo-600'
    },
    orange: {
      bg: 'bg-orange-50',
      border: 'border-orange-200',
      titleColor: 'text-orange-900',
      iconColor: 'text-orange-600'
    }
  };

  const colors = colorSchemes[colorScheme] || colorSchemes.gray;

  return (
    <div className={`mt-8 ${colors.bg} rounded-xl border ${colors.border} p-6 ${className}`}>
      {title && (
        <h3 className={`text-lg font-semibold ${colors.titleColor} mb-4 flex items-center`}>
          {icon && (
            <span className={`mr-2 ${colors.iconColor}`}>
              {icon}
            </span>
          )}
          {title}
        </h3>
      )}
      {children}
    </div>
  );
});

export default ResultCard;
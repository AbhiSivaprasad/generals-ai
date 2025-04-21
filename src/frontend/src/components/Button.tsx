import React from 'react';

interface ButtonProps {
    children: React.ReactNode;
    size?: 'small' | 'medium' | 'large';
    onClick?: () => void;
    className?: string;
}

const Button: React.FC<ButtonProps> = ({
    children,
    size = 'medium',
    onClick,
    className = ''
}) => {
    const baseClasses = 'font-semibold rounded-lg transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-opacity-50';

    const sizeClasses = {
        small: 'px-2 py-1 text-sm',
        medium: 'px-4 py-2 text-base',
        large: 'px-6 py-3 text-lg'
    };

    const hoverClasses = 'hover:bg-opacity-80 hover:shadow-md';

    const classes = `${baseClasses} ${sizeClasses[size]} ${hoverClasses} ${className}`;

    return (
        <button
            className={classes}
            onClick={onClick}
        >
            {children}
        </button>
    );
};

export default Button;

import React, { InputHTMLAttributes, useEffect } from 'react';
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react';

type InputSize = 'small' | 'medium' | 'large';
type InputState = 'loading' | 'success' | 'error' | undefined;

interface CustomInputProps extends InputHTMLAttributes<HTMLInputElement> {
    inputSize?: InputSize;
    state?: InputState;
    label?: string;
    onEnter?: () => void;
}

const Input: React.FC<CustomInputProps> = ({
    inputSize = 'medium',
    state,
    onEnter,
    className = '',
    label,
    ...props
}) => {
    const sizeClasses = {
        small: 'px-2 py-1 text-sm',
        medium: 'px-3 py-2 text-base',
        large: 'px-4 py-3 text-lg',
    } as const;

    const stateClasses = {
        loading: 'border-blue-300 focus:border-blue-500 focus:ring-blue-500',
        success: 'border-green-300 focus:border-green-500 focus:ring-green-500',
        error: 'border-red-300 focus:border-red-500 focus:ring-red-500',
        undefined: 'border-gray-300 focus:border-blue-500 focus:ring-blue-500',
    };

    const StateIcon = () => {
        switch (state) {
            case 'loading':
                return <Loader2 className="animate-spin text-blue-500" size={inputSize === 'small' ? 16 : inputSize === 'large' ? 24 : 20} />;
            case 'success':
                return <CheckCircle className="text-green-500" size={inputSize === 'small' ? 16 : inputSize === 'large' ? 24 : 20} />;
            case 'error':
                return <AlertCircle className="text-red-500" size={inputSize === 'small' ? 16 : inputSize === 'large' ? 24 : 20} />;
            default:
                return null;
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (onEnter && e.key === "Enter") {
            onEnter()
        }
    }

    return (
        <div className="relative">
            {label && (
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    {label}
                </label>
            )}
            <input
                onKeyDown={handleKeyDown}
                className={`
          w-full rounded-md border 
          ${sizeClasses[inputSize]}
          ${stateClasses[state || 'undefined']}
          focus:outline-none focus:ring-2 focus:ring-opacity-50
          transition duration-150 ease-in-out
          ${className}
        `}
                {...props}
            />
            {state && (
                <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    <StateIcon />
                </div>
            )}
        </div>
    );
};

export default Input;
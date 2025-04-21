import React, { useState, useCallback, ReactNode } from 'react';

interface ModalProps {
    children: ReactNode;
}

const Modal: React.FC<ModalProps> = ({ children }) => {

    return (
        <>
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
                    <div className="mb-4">{children}</div>
                </div>
            </div>
        </>
    );
};

export default Modal;
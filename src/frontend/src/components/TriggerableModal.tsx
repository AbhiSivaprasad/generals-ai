import React, { useState, useCallback, ReactNode } from 'react';

interface TriggerableModalProps {
    children: ReactNode;
    content: ReactNode;
}

const TriggerableModal: React.FC<TriggerableModalProps> = ({ children, content }) => {
    const [isOpen, setIsOpen] = useState(false);

    const openModal = useCallback(() => setIsOpen(true), []);
    const closeModal = useCallback(() => setIsOpen(false), []);

    return (
        <>
            <div onClick={openModal}>{children}</div>
            {isOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
                        <div className="mb-4">{content}</div>
                        <button
                            onClick={closeModal}
                            className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
        </>
    );
};

export default TriggerableModal;
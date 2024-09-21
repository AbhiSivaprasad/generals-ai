"use client";

import React, { useEffect, useState } from 'react';
import socket, { BoardStateMessage } from '../../socketio/client';
import Router, { useRouter } from 'next/navigation';

const WaitingPage: React.FC = () => {
    const [dots, setDots] = useState('');

    const router = useRouter();

    useEffect(() => {
        // Join the game queue when the component mounts
        socket.emit('join-game', { playBot: true });

        socket.on('game_started', (board_state: BoardStateMessage) => {
            // Save the board state to local storage
            localStorage.setItem('board_state', JSON.stringify(board_state));
            router.push(`/game/${board_state.game_id}`);
        })

        // Animate the loading dots
        const interval = setInterval(() => {
            setDots(prevDots => (prevDots.length >= 3 ? '' : prevDots + '.'));
        }, 500);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <h1 className="text-4xl font-bold mb-4 text-gray-800">Waiting for game</h1>
            <p className="text-xl text-gray-600">
                Please wait while we find a match for you{dots}
            </p>
            <div className="mt-8">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        </div>
    );
};

export default WaitingPage;

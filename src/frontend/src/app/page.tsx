'use client';

import React, { useEffect, useState } from 'react';
import Button from '../components/Button';
import socket from '../socketio/client';
import Input from '../components/Input';
import { useRouter } from 'next/navigation';
import TriggerableModal from '@/components/TriggerableModal';
import { PencilIcon } from 'lucide-react';
import Modal from '@/components/Modal';

const App: React.FC = () => {

    const router = useRouter();
    const [username, setUsername] = useState(localStorage.getItem('username'))
    const [usernameInput, setUsernameInput] = useState(username ?? "")
    const [isEditingUsername, setIsEditingUsername] = useState(false)

    useEffect(() => {
        socket.on('connect', () => {
            console.log('Connected to server');
        });
    }, []);

    // Sync the username both with localhost to save between sessions,
    // And to the server so that it can use it.
    useEffect(() => {
        if (username) {
            localStorage.setItem('username', username)
            socket.emit("set_username", username)
        }
    }, [username])

    const [replayQuery, setReplayQuery] = useState('');
    const handleViewReplays = () => {
        // TODO: Implement view replays functionality
        console.log('View replays clicked');
    };

    const handlePlayAgainstBot = () => {
        // TODO: Implement play against bot functionality
        router.push('/game');
    };

    const handlePlayAgainstPlayer = () => {
        // TODO: Implement play against player functionality
        console.log('Play against player clicked');

    };

    const saveUsername = () => {
        setUsername(usernameInput)
        setIsEditingUsername(false)
    }

    const replaysPopupContent = <div><Input label="Replay ID" placeholder="1234" value={replayQuery} onChange={(e) => setReplayQuery(e.target.value)} inputSize="large" /></div>

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <div className="text-4xl font-bold mb-8">generals.ai</div>
            {username ? <div className='flex items-center gap-2 cursor-pointer' onClick={() => setIsEditingUsername(true)}>{username} <PencilIcon className='w-4 h-4' /></div> : <div className="cursor-pointer" onClick={() => setIsEditingUsername(true)}>No username, click to set</div>}
            {isEditingUsername && <Modal>
                <Input onEnter={saveUsername} value={usernameInput} onChange={(e) => setUsernameInput(e.target.value)} />
                <Button onClick={saveUsername}>Save</Button>
            </Modal>}

            <div className="space-y-4 space-x-4 flex flex-row items-center align-center">
                <TriggerableModal content={replaysPopupContent}>
                    <Button className="bg-blue-500 text-white w-48">
                        View Replays
                    </Button>
                </TriggerableModal>
                <Button onClick={handlePlayAgainstBot} className="bg-green-500 text-white w-48">
                    Play Against Bot
                </Button>
                <Button onClick={handlePlayAgainstPlayer} className="bg-purple-500 text-white w-48">
                    Play Against Player
                </Button>
            </div>
        </div>
    );
};

export default App;

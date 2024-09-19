import React from 'react';
import Button from './Button';

const App: React.FC = () => {
    const handleViewReplays = () => {
        // TODO: Implement view replays functionality
        console.log('View replays clicked');
    };

    const handlePlayAgainstBot = () => {
        // TODO: Implement play against bot functionality
        console.log('Play against bot clicked');
    };

    const handlePlayAgainstPlayer = () => {
        // TODO: Implement play against player functionality
        console.log('Play against player clicked');
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <h1 className="text-4xl font-bold mb-8">generals.ai</h1>
            <div className="space-y-4">
                <Button onClick={handleViewReplays} className="bg-blue-500 text-white w-48">
                    View Replays
                </Button>
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

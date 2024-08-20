import React from 'react';
import Button from './Button';

<<<<<<< HEAD
const App: React.FC = () => {
    const handleViewReplays = () => {
        // TODO: Implement view replays functionality
        console.log('View replays clicked');
    };
=======
// custom font
import "typeface-quicksand"
import { useData } from './api/utils';
import Replay from './replay';
import { CellState, BoardState } from './types/globals';
import { useParams, useMatch } from 'react-router-dom';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

>>>>>>> origin/master

    const handlePlayAgainstBot = () => {
        // TODO: Implement play against bot functionality
        console.log('Play against bot clicked');
    };

<<<<<<< HEAD
    const handlePlayAgainstPlayer = () => {
        // TODO: Implement play against player functionality
        console.log('Play against player clicked');
    };

=======
function App() {
    return (
        <Router>
            <Routes>
                <Route path="/replay/*" element={<InnerApp/>} />
            </Routes>
        </Router>
    );
}


function InnerApp() {
    const [playerIndex, setPlayerIndex] = useState<number | null>(null);
    const [boardStates, setBoardStates] = useState<any[]>([]);

    const match = useMatch('/replay/*');
    const replayPath = match?.pathname.split('/replay/')[1] || '';
    const replayData = useData(`/replay/${replayPath}`);

    useEffect(() => {
        if(!replayData) return;

        const data = replayData?.data;
        console.log(data);

        // let boardStates;
        // let viewStatesList = new Array(data.numViews);

        const initialBoardState = data.initialBoard as BoardState;
        const diffs = data.boardDiffs;
        var allBoardStates: BoardState[] = [initialBoardState];

        var newBoardState = initialBoardState;
        for(const diff of diffs) {
            newBoardState = JSON.parse(JSON.stringify(newBoardState));
            patch(newBoardState, diff);
            allBoardStates = [...allBoardStates, newBoardState];
        }

        setBoardStates(allBoardStates);
    }, [replayData]);


    const onViewButtonClick = (e: any) => {
        setPlayerIndex(playerIndex === e.target.value ? null : e.target.value)
    }
>>>>>>> origin/master
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

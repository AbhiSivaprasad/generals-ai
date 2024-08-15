import React, { useEffect, useState } from 'react';

// custom font
import "typeface-quicksand"
import { useData } from './api/utils';
import Replay from './replay';
import { CellState, BoardState } from './types/globals';
import { useParams } from 'react-router-dom';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';


/**
 * apply a move to a board state and return new board state
 * @param board
 * @param diffs: list of diff
 * @returns BoardState
 */
function patch(board: BoardState, diffs: CellState[]) {
    for(const diff of diffs) {
        board[diff.y][diff.x] = diff;
    }
    return board;
}

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/replay/:replayId" element={<InnerApp/>} />
            </Routes>
        </Router>
    );
}


function InnerApp() {
    const [playerIndex, setPlayerIndex] = useState<number | null>(null);
    const [boardStates, setBoardStates] = useState<any[]>([]);
    const { replayId } = useParams<{ replayId: string }>();

    const replayData = useData(`/replay/${replayId}`);

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
    return (
        <div>
            <Replay boardStates={boardStates} playerIndex={playerIndex} />
            <button onClick={onViewButtonClick} value={0}>blue</button>
            <button onClick={onViewButtonClick} value={1}>red</button>
        </div>
    );
}

export default App;

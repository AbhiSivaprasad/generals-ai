import React, { useEffect, useState } from 'react';

// custom font
import "typeface-quicksand"
import { useData } from './api/utils';
import Replay from './replay';


function computeNextBoardState(prevBoardState: string | any[], boardDiffs: any, time: number) {
    // initialize first dimension of two dimensional array
    let nextBoardState = new Array(prevBoardState.length);

    for(let i = 0; i < prevBoardState.length; i++) {
        // initialize second dimension
        nextBoardState[i] = new Array(prevBoardState[0].length);
        for (let j = 0; j < prevBoardState[0].length; j++)
            nextBoardState[i][j] = Object.assign({}, prevBoardState[i][j]);
    }

    // patch moves
    patch(nextBoardState, boardDiffs);

    // increment cities and generals every tick and land every 25 ticks
    for(let i = 0; i < prevBoardState.length; i++) {
        for(let j = 0; j < prevBoardState.length; j++) {
            // update cell state
            let cell = nextBoardState[i][j];
            if(time % 2 === 0
                && (cell.isGeneral  // player's general
                || (cell.isCity && cell.type >= 0) // player's city
                || (cell.type >= 0 && time % (25 * 2) === 0))) {  // player's land tile every 25 ticks
                cell.army += 1;
            }
        }
    }

    return nextBoardState
}

function computeAllBoardStates(initialState: any, diffsList: string | any[]) {
    let boardStates = [];
    let boardState = initialState;

    boardStates.push(boardState);
    for(let i = 0; i < diffsList.length; i++) {
        boardState = computeNextBoardState(boardState, diffsList[i], i);
        boardStates.push(boardState);
    }

    return boardStates
}

/**
 * apply a move to a board state and return new board state
 * @param board
 * @param diffs: list of diff
 * @returns {*}
 */
function patch(board: { [y: number]: { [x: number]: any; }; }, diffs: string | any[]) {
    for(let i = 0; i < diffs.length; i++) {
        const diff = diffs[i];
        let updateCell = board[diff.y][diff.x];

        updateCell.army = diff.army;
        updateCell.type = diff.type;
        updateCell.isCity = diff.isCity;
        updateCell.isGeneral = diff.isGeneral;
        updateCell.memory = diff.memory;
    }
}


function App() {
    const [playerIndex, setPlayerIndex] = useState<number | null>(null);
    const [viewStatesList, setViewStatesList] = useState<any[]>([]);
    const [boardStates, setBoardStates] = useState<any[]>([]);

    const replayData = useData("/replay");

    useEffect(() => {
        if(!replayData) return;
        console.log(replayData);

        const data = replayData?.data;

        let boardStates;
        let viewStatesList = new Array(data.numViews);

        let initialBoardState = data.gameLog.initialBoard;
        let boardDiffsList = data.gameLog.boardDiffs;

        boardStates = computeAllBoardStates(initialBoardState, boardDiffsList);

        for(let i = 0; i < data.numViews; i++) {
            let initialViewState = data.viewLogs[i].initialBoard;
            let viewDiffsList = data.viewLogs[i].boardDiffs;

            viewStatesList[i] = computeAllBoardStates(initialViewState, viewDiffsList);
        }

        // update state with parsed data
        setBoardStates(boardStates);
        setViewStatesList(viewStatesList);

    }, [replayData]);


    const onViewButtonClick = (e: any) => {
        console.log(viewStatesList);
        setPlayerIndex(playerIndex === e.target.value ? null : e.target.value)
    }
    return (
        <div>
            <Replay boardStates={
                playerIndex == null ? boardStates : viewStatesList[playerIndex]
            } />
            <button onClick={onViewButtonClick} value="0">blue</button>
            <button onClick={onViewButtonClick} value="1">red</button>
        </div>
    );
}

export default App;

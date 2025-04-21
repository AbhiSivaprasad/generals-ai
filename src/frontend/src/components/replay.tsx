import React, { Component, useState } from 'react';
import Board from './board.js'
import PlayBar from './playbar.js'

// custom font
import "typeface-quicksand"
import { BoardDiff, BoardState } from './types/globals.js';

interface ReplayProps {
    boardStates: any[][];
    playerIndex: number
}


const Replay: React.FC<ReplayProps> = ({ boardStates, playerIndex }) => {
    const [time, setTime] = useState(0);
    const [autoPlaySpeed, setAutoPlaySpeed] = useState(1);
    const [interval, setInterval] = useState<NodeJS.Timeout | null>(null);

    const onTick = () => {
        if (time < boardStates.length - 1) {
            // increment time
            setTime(time + 1)
        } else {
            // finished replay
            if (interval != null) {
                clearInterval(interval);
                setInterval(null);
            }
        }
    }

    const computeAllBoardStates = (initialState: BoardState, boardDiffsList: BoardDif[]) => {
        let boardStates = [];
        let boardState = initialState;

        // const initialState = [[{'type': -1, 'army': 0, 'isCity': false, 'isGeneral': false}, {'type': 0, 'army': 1, 'isCity': false, 'isGeneral': true}], [{'type': 1, 'army': 1, 'isCity': false, 'isGeneral': true}, {'type': -1, 'army': 0, 'isCity': false, 'isGeneral': false}]]

        boardStates.push(boardState);
        for (let i = 0; i < boardDiffsList.length; i++) {
            boardState = this.computeNextBoardState(boardState, boardDiffsList[i]);
            boardStates.push(boardState);
        }

        return boardStates
    }

    computeNextBoardState(prevBoardState, boardDiffs) {
        // initialize first dimension of two dimensional array
        let nextBoardState = new Array(prevBoardState.length);

        // increment cities and generals every tick and land every 25 ticks
        for (let i = 0; i < prevBoardState.length; i++) {
            // initialize second dimension
            nextBoardState[i] = new Array(prevBoardState[0].length);
            for (let j = 0; j < prevBoardState.length; j++) {
                nextBoardState[i][j] = Object.assign({}, prevBoardState[i][j]);

                // update cell state
                const cell = nextBoardState[i][j];
                if (cell.isGeneral // player's general
                    || (cell.type === "city") // player's city
                    || (cell.type !== "normal" && this.state.time % (25 * 2) === 0)) {  // player's land tile every 25 ticks
                    cell.army += 1;
                }
            }
        }

        // patch moves
        Replay.patch(nextBoardState, boardDiffs);

        return nextBoardState
    }

    /**
     * Generate an onClick method for autoplay buttons given a particular speed
     * @param autoPlaySpeed
     */
    onAutoPlaySpeedClickFactory(autoPlaySpeed) {
        return (e) => {
            // only execute if new speed pressed
            if (this.state.autoPlaySpeed === autoPlaySpeed)
                return;

            // check if currently playing
            if (this.state.interval != null) {
                // clear current interval
                clearInterval(this.state.interval);

                const interval = setInterval(this.onTick, 1000 / autoPlaySpeed);

                this.setState({
                    autoPlaySpeed: autoPlaySpeed,
                    interval: interval
                })
            } else {
                // not currently playing so don't mess with interval
                this.setState({
                    autoPlaySpeed: autoPlaySpeed
                })
            }
        };
    }

    /**
     * onClick handler for play button
     * @param e
     */
    onPlayButtonClick(e) {
        // only play if not already playing
        if (this.state.interval == null) {
            // if replay is over and play is clicked then restart the replay
            if (this.state.time === this.props.boardStates.length - 1)
                this.setState({ time: 0 });

            const interval = setInterval(this.onTick, 1000 / this.state.autoPlaySpeed);
            this.setState({ interval: interval })
        }
    }

    /**
     * onClick handler for pause button
     * @param e
     */
    onPauseButtonClick(e) {
        // only pause if currently playing
        if (this.state.interval != null) {
            clearInterval(this.state.interval);
            this.setState({ interval: null });
        }
    }

    /**
     * onClick handler for rewinding replay one move
     * @param e
     */
    onLastMoveClick(e) {
        let stateUpdate = {};

        // stop autoplay if currently on
        if (this.state.interval != null) {
            clearInterval(this.state.interval);
            stateUpdate["interval"] = null;
        }

        // if not at first state then move back one state
        if (this.state.time !== 0)
            stateUpdate["time"] = this.state.time - 1;

        // update state
        this.setState(stateUpdate);
    }

    /**
     * onClick handler for next move in replay
     * @param e
     */
    onNextMoveClick(e) {
        let stateUpdate = {};

        // stop autoplay if currently on
        if (this.state.interval != null) {
            clearInterval(this.state.interval);
            stateUpdate["interval"] = null;
        }

        // if not at first state then move back one state
        if (this.state.time !== this.props.boardStates.length - 1)
            stateUpdate["time"] = this.state.time + 1;

        // update state
        this.setState(stateUpdate);
    }



    return boardStates.length !== 0 && (
        <div>
            <Board data={boardStates[time]} playerIndex={playerIndex} />
            <PlayBar onAutoPlaySpeedClickFactory={onAutoPlaySpeedClickFactory}
                onPlayButtonClick={onPlayButtonClick}
                onPauseButtonClick={onPauseButtonClick}
                onLastMoveClick={onLastMoveClick}
                onNextMoveClick={onNextMoveClick} />
            turn: {time}
        </div>
    )
}

export default Replay;

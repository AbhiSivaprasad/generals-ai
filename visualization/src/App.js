import React, { Component } from 'react';
import Replay from "./replay.js"

// custom font
import "typeface-quicksand"


class App extends Component {
    constructor() {
        super();

        this.state = {
            playerIndex: null,
            viewStatesList: [],
            boardStates: []
        };

        this.onViewButtonClick = this.onViewButtonClick.bind(this)
    }

    render() {
        return (
            <div>
                <Replay boardStates={
                    this.state.playerIndex == null
                        ? this.state.boardStates : this.state.viewStatesList[this.state.playerIndex]
                } />
                <button onClick={this.onViewButtonClick} value="0">blue</button>
                <button onClick={this.onViewButtonClick} value="1">red</button>
            </div>
        )
    }

    componentDidMount() {
        fetch(`/replay`)
            .then(response => response.json())
            .then((data) => {
                let boardStates;
                let viewStatesList = new Array(data.numViews);

                let initialBoardState = data.gameLog.initialBoard;
                let boardDiffsList = data.gameLog.boardDiffs;

                boardStates = this.computeAllBoardStates(initialBoardState, boardDiffsList);

                for(let i = 0; i < data.numViews; i++) {
                    let initialViewState = data.viewLogs[i].initialBoard;
                    let viewDiffsList = data.viewLogs[i].boardDiffs;

                    viewStatesList[i] = this.computeAllBoardStates(initialViewState, viewDiffsList);
                }

                // update state with parsed data
                this.setState({boardStates: boardStates, viewStatesList: viewStatesList})
            })
    }

    computeAllBoardStates(initialState, diffsList) {
        let boardStates = [];
        let boardState = initialState;

        // const initialState = [[{'type': -1, 'army': 0, 'isCity': false, 'isGeneral': false}, {'type': 0, 'army': 1, 'isCity': false, 'isGeneral': true}], [{'type': 1, 'army': 1, 'isCity': false, 'isGeneral': true}, {'type': -1, 'army': 0, 'isCity': false, 'isGeneral': false}]]

        boardStates.push(boardState);
        for(let i = 0; i < diffsList.length; i++) {
            boardState = this.computeNextBoardState(boardState, diffsList[i], i);
            boardStates.push(boardState);
        }

        return boardStates
    }

    computeNextBoardState(prevBoardState, boardDiffs, time) {
        // initialize first dimension of two dimensional array
        let nextBoardState = new Array(prevBoardState.length);

        for(let i = 0; i < prevBoardState.length; i++) {
            // initialize second dimension
            nextBoardState[i] = new Array(prevBoardState[0].length);
            for (let j = 0; j < prevBoardState[0].length; j++)
                nextBoardState[i][j] = Object.assign({}, prevBoardState[i][j]);
        }

        // patch moves
        this.patch(nextBoardState, boardDiffs);

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

    /**
     * apply a move to a board state and return new board state
     * @param board
     * @param diffs: list of diff
     * @returns {*}
     */
    patch(board, diffs) {
        for(let i = 0; i < diffs.length; i++) {
            const diff = diffs[i];
            let updateCell = board[diff.y][diff.x];

            updateCell.army = diff.army;
            updateCell.type = diff.type;
            updateCell.isCity = diff.isCity;
            updateCell.isGeneral = diff.isGeneral;
        }
    }

    onViewButtonClick(e) {
        console.log(this.state.viewStatesList);
        this.setState({playerIndex: this.state.playerIndex === e.target.value ? null : e.target.value})
    }
}

export default App;

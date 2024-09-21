import React from 'react';
import Row from "./row"
import { Action, CellState } from '../app/types/globals';

interface BoardProps {
    data: CellState[][];
    playerIndex: number;
    selectedCell: [number, number] | null;
    fogOfWarEnabled: boolean;
    onCellSelect: (rowIndex: number, columnIndex: number) => void;
    moveQueue: Action[];
    zoom: number;
}

const Board: React.FC<BoardProps> = ({ data, playerIndex, fogOfWarEnabled, selectedCell, onCellSelect, moveQueue, zoom }) => {
    const tableStyle = {
        borderCollapse: "collapse",
        border: "1px solid black",
        position: "relative"
    } as const;

    const getArrowPosition = (move: Action) => {
        const cellSize = 40 * zoom; // Adjust this value based on your cell size
        let top = move.rowIndex * cellSize;
        let left = move.columnIndex * cellSize;
        let arrowChar = '';

        switch (move.direction) {
            case 'up':
                top = top - cellSize / 2;
                left = left + cellSize / 4;
                arrowChar = '↑';
                break;
            case 'down':
                top = top + cellSize / 2;
                left = left + cellSize / 4;
                arrowChar = '↓';
                break;
            case 'left':
                left = left - cellSize / 3.5;
                arrowChar = '←';
                break;
            case 'right':
                left = left + cellSize / 1.5;
                arrowChar = '→';
                break;
        }

        return { top, left, arrowChar };
    };

    return (
        <div style={{ position: 'relative' }}>
            <table style={tableStyle}>
                <tbody>
                    {
                        data.map((item, index) => (
                            <Row fogOfWarEnabled={fogOfWarEnabled} playerIndex={playerIndex} zoom={zoom} data={item} key={index} selectedCell={selectedCell && selectedCell[0] == index ? selectedCell[1] : null} onCellSelect={(columnIndex) => onCellSelect(index, columnIndex)} />
                        ))
                    }
                </tbody>
            </table>
            {
                moveQueue.map((move, index) => {
                    const { top, left, arrowChar } = getArrowPosition(move);
                    return (
                        <div key={index} style={{
                            position: 'absolute',
                            top: `${top}px`,
                            left: `${left}px`,
                            fontSize: '24px',
                            color: 'white',
                            textShadow: '1,1',
                            pointerEvents: 'none'
                        }}>
                            {arrowChar}
                        </div>
                    );
                })
            }
        </div>
    );
};

export default Board;
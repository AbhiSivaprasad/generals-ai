import React from 'react';
import { CellState } from '../app/types/globals';
import Cell from './cell';

interface RowProps {
    data: CellState[];
    onCellSelect: (columnIndex: number) => void;
    zoom: number;
    playerIndex: number;
    selectedCell: number | null;
    fogOfWarEnabled: boolean;
}

const Row: React.FC<RowProps> = ({ data, onCellSelect, selectedCell, zoom, playerIndex, fogOfWarEnabled }) => {
    return (
        <tr className="row">
            {
                data.map((item, index) => (
                    <Cell fogOfWarEnabled={fogOfWarEnabled} playerIndex={playerIndex} data={item} key={index} onSelect={() => onCellSelect(index)} selected={selectedCell == index} zoom={zoom} />
                ))
            }
        </tr>
    );
};

export default Row;

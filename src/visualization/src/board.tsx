import React from 'react';
import Row from './row';

interface BoardProps {
    data: any[];
    playerIndex: number;
    fogOfWarEnabled: boolean
}

const Board: React.FC<BoardProps> = ({ data, playerIndex, fogOfWarEnabled }) => {
    const tableStyle = {
        borderCollapse: "collapse",
        border: "1px solid black"
    } as const;

    return (
        <table style={tableStyle}>
            <tbody>
                {
                    data.map((item, index) => (
                        <Row fogOfWarEnabled={fogOfWarEnabled} data={item} key={index} playerIndex={playerIndex} />
                    ))
                }
            </tbody>
        </table>
    );
};

export default Board;
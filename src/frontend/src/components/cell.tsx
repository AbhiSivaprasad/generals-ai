import React from 'react';
import Image from 'next/image';
import { CellState } from '../app/types/globals';


interface CellProps {
    data: CellState;
    selected: boolean;
    onSelect: () => void;
    zoom: number;
    playerIndex: number;
    fogOfWarEnabled: boolean;
}

const CELL_SIZE_AT_1_ZOOM_PX = 40;

const Cell: React.FC<CellProps> = ({ data, onSelect, selected, zoom, playerIndex, fogOfWarEnabled }) => {
    const cellStyle: React.CSSProperties = {
        border: "1px solid black",
        width: `${CELL_SIZE_AT_1_ZOOM_PX * zoom}px`,
        height: `${CELL_SIZE_AT_1_ZOOM_PX * zoom}px`,
        whiteSpace: "nowrap",
        color: "white",
        fontSize: "12px",
        textAlign: "center",
        textShadow: "0 0 2px black",
        fontFamily: "Quicksand",
        textOverflow: "ellipsis",
        borderWidth: "1px",
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        backgroundSize: "80%",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        outline: selected ? "2px solid white" : "none",
        position: "relative",
    };

    const playerColors = ["blue", "red"];
    const cellOwnerPlayerIndex = data.player_index;

    let backgroundImage = '';

    const isVisible = !fogOfWarEnabled || cellOwnerPlayerIndex === playerIndex || data.player_visibilities[playerIndex]

    switch (data.type) {
        case "normal":
            if (isVisible) {
                cellStyle.backgroundColor = data.player_index !== null ? playerColors[data.player_index] : "#dcdcdc";
            }
            break;
        case "general":
            if (isVisible) {
                backgroundImage = '/images/crown.png';
                if (data.player_index !== null) {
                    cellStyle.backgroundColor = playerColors[data.player_index];
                } else {
                    console.error("GENERAL CELL HAS NO PLAYER INDEX!");
                }
            }
            break;
        case "mountain":
            backgroundImage = '/images/mountain.png';
            if (isVisible) {
                cellStyle.backgroundColor = "#bbb";
            }
            break;
        case "city":
            if (!isVisible) {
                backgroundImage = '/images/mountain.png'
            } else {
                backgroundImage = '/images/city.png';
                if (data.player_index !== null) {
                    cellStyle.backgroundColor = playerColors[data.player_index];
                }
            }
            break;
        default:
            break;
    }

    return (
        <td style={cellStyle} onClick={onSelect}>
            {backgroundImage && (
                <Image
                    src={backgroundImage}
                    alt={data.type}
                    width={24}
                    height={24}
                    style={{ opacity: 0.7, position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
                />
            )}
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                {(isVisible && data.army !== 0) && data.army}
            </div>
        </td>
    );
};

export default Cell;

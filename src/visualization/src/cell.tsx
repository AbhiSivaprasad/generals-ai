import React from 'react';
import CityBackground from './resources/city.png';
import CrownBackground from './resources/crown.png';
import MountainBackground from './resources/mountain.png';

interface CellProps {
    data: {
        type: string;
        player_visibilities: boolean[];
        player_index: number | null;
        army: number;
    };
    playerIndex: number;
}

const Cell: React.FC<CellProps> = ({ data, playerIndex }) => {
    const cellStyle: React.CSSProperties = {
        border: "1px solid black",
        width: "32px",
        height: "32px",
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
    };

    const playerColors = ["blue", "red"];

    switch (data.type) {
        case "normal":
            if (!playerIndex || data.player_visibilities[playerIndex]) {
                cellStyle.backgroundColor = data.player_index !== null ? playerColors[data.player_index] : "#dcdcdc";
            }
            break;
        case "general":
            if (!playerIndex || data.player_visibilities[playerIndex]) {
                cellStyle.backgroundImage = `url(${CrownBackground})`;
                if (data.player_index !== null) {
                    cellStyle.backgroundColor = playerColors[data.player_index];
                } else {
                    console.error("GENERAL CELL HAS NO PLAYER INDEX!");
                }
            }
            break;
        case "mountain":
            cellStyle.backgroundImage = `url(${MountainBackground})`;
            if (!playerIndex || data.player_visibilities[playerIndex]) {
                cellStyle.backgroundColor = "#bbb";
            }
            break;
        case "city":
            cellStyle.backgroundImage = `url(${CityBackground})`;
            if (!playerIndex || data.player_visibilities[playerIndex]) {
                cellStyle.backgroundColor = data.player_index !== null ? playerColors[data.player_index] : "gray";
            }
            break;
        default:
            break;
    }

    return (
        <td style={cellStyle}>
            {((!playerIndex || data.player_visibilities[playerIndex]) && data.army !== 0) && data.army}
        </td>
    );
};

export default Cell;

import React, { Component } from 'react';
import CityBackground from './resources/city.png';
import CrownBackground from './resources/crown.png';
import MountainBackground from './resources/mountain.png';

class Cell extends Component {
    render() {
        let cellStyle = {
            border: "1px solid black",
            width: "32px",
            height: "32px",
            whitespace: "nowrap",
            color: "white",
            fontSize: "12px",
            textAlign: "center",
            textShadow: "0 0 2px black",
            fontFamily: "Quicksand",
            textOverflow: "ellipsis",
            borderWidth: "1px"
        };

        const playerColors = ["blue", "red"];

        cellStyle["backgroundColor"] = "rgba(0, 0, 0, 0.6)";
        cellStyle["borderWidth"] = 0;

        const cell = this.props.data;

        switch (cell.type) {
            case "normal":
                if(!this.props.playerIndex || cell.player_visibilities[this.props.playerIndex]) {
                    if(cell.player_index != null) {
                        cellStyle["backgroundColor"] = playerColors[cell.player_index];
                    }
                    else {
                        cellStyle["backgroundColor"] = "#dcdcdc";
                    }
                }
                break;
            case "general":
                if(!this.props.playerIndex || cell.player_visibilities[this.props.playerIndex]) {
                    cellStyle["backgroundImage"] = `url(${CrownBackground})`;
                    if(cell.player_index != null) {
                        cellStyle["backgroundColor"] = playerColors[cell.player_index];
                    }
                    else {
                        console.error("GENERAL CELL HAS NO PLAYER INDEX!");
                    }
                }
                break;
            case "mountain":
                cellStyle["backgroundImage"] = `url(${MountainBackground})`;
                if(!this.props.playerIndex || cell.player_visibilities[this.props.playerIndex]) {
                    cellStyle["backgroundColor"] = "#bbb";
                }
                break;
            case "city":
                // visible empty (city, no city)
                cellStyle["backgroundImage"] = `url(${MountainBackground})`;
                if(!this.props.playerIndex || cell.player_visibilities[this.props.playerIndex]) {
                    cellStyle["backgroundImage"] = `url(${CityBackground})`;
                    if(cell.player_index != null) {
                        cellStyle["backgroundColor"] = playerColors[cell.player_index];
                    }
                    else {
                        cellStyle["backgroundColor"] = "gray";
                    }
                }
                break;
            default:
                break;
        }

        // if background image was added then set background style to 100%
        if("backgroundImage" in cellStyle) {
            cellStyle["backgroundSize"] = "80%";
            cellStyle["backgroundRepeat"] = "no-repeat";
            cellStyle["backgroundPosition"] = "center";
        }

        return (
            <td style={cellStyle}>
                {this.props.data.army !== 0 ? this.props.data.army : null}
            </td>
        )
    }
}

export default Cell

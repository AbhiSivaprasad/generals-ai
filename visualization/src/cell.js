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
        switch (this.props.data.type) {
            case -4:
            case -3:
                // fog obstacle or no obstacle
                cellStyle["backgroundColor"] = "rgba(0, 0, 0, 0.6)";
                cellStyle["borderWidth"] = 0;
                break;
            case -2:
                // visible mountain
                cellStyle["backgroundColor"] = "#bbb";
                break;
            case -1:
                // visible empty (city, no city)
                cellStyle["backgroundColor"] = this.props.data.isCity ? "gray" : "#dcdcdc";
                break;
            default:
                // player index
                cellStyle["backgroundColor"] = playerColors[this.props.data.type];
        }

        // set background image for cell
        if(this.props.data.isCity) {
            cellStyle["backgroundImage"] = `url(${CityBackground})`;
        } else if(this.props.data.isGeneral) {
            cellStyle["backgroundImage"] = `url(${CrownBackground})`;
        } else if(this.props.data.type === -2 || this.props.data.type === -4) {
            cellStyle["backgroundImage"] = `url(${MountainBackground})`;
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

import React, { Component } from 'react';
import Row from "./row.js"

class Board extends Component {
    render() {
        const tableStyle = {
            borderCollapse: "collapse",
            border: "1px solid black"
        };

        return (
            <table style={tableStyle}>
                <tbody>
                    {
                        this.props.data.map((item, index) => {
                            return <Row data={item} key={index} />
                        })
                    }
                </tbody>
            </table>
        )
    }
}

export default Board;
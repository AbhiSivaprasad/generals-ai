import React, { Component } from 'react';
import Cell from "./cell.js"

class Row extends Component {
    render() {
        return (
            <tr className="row">
                {
                    this.props.data.map((item, index) => {
                        return <Cell data={item} key={index} />
                    })
                }
            </tr>
        )
    }
}

export default Row

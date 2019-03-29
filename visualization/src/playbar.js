import React, { Component } from 'react';
import PauseBackground from './resources/pause.png'
import PlayBackground from './resources/play.png'

class PlayBar extends Component {
    render() {
        return (
            <div>
                <div>
                    <button onClick={this.props.onLastMoveClick}>
                        <span>←</span>
                        <br/>
                        <span>Last Move</span>
                    </button>
                    <img
                        src={PlayBackground}
                        onClick={this.props.onPlayButtonClick}
                        height="30px"
                        width="30px"
                    />
                    <img
                        src={PauseBackground}
                        onClick={this.props.onPauseButtonClick}
                        height="30px"
                        width="30px"
                    />
                    <button onClick={this.props.onNextMoveClick}>
                        <span>→</span>
                        <br/>
                        <span>Next Move</span>
                    </button>
                </div>
                <div>
                    <button onClick={this.props.onAutoPlaySpeedClickFactory(1)}>1x</button>
                    <button onClick={this.props.onAutoPlaySpeedClickFactory(2)}>2x</button>
                    <button onClick={this.props.onAutoPlaySpeedClickFactory(5)}>5x</button>
                    <button onClick={this.props.onAutoPlaySpeedClickFactory(10)}>10x</button>
                    <button onClick={this.props.onAutoPlaySpeedClickFactory(50)}>50x</button>
                </div>
            </div>
        )
    }
}

export default PlayBar
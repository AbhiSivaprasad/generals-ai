import React from 'react';
import PauseBackground from './resources/pause.png'
import PlayBackground from './resources/play.png'

interface PlayBarProps {
    onLastMoveClick: () => void;
    onPlayButtonClick: () => void;
    onPauseButtonClick: () => void;
    onNextMoveClick: () => void;
    onAutoPlaySpeedClickFactory: (speed: number) => () => void;
}

const PlayBar: React.FC<PlayBarProps> = ({
    onLastMoveClick,
    onPlayButtonClick,
    onPauseButtonClick,
    onNextMoveClick,
    onAutoPlaySpeedClickFactory
}) => {
    return (
        <div>
            <div>
                <button onClick={onLastMoveClick}>
                    <span>←</span>
                    <br />
                    <span>Last Move</span>
                </button>
                <img
                    src={PlayBackground}
                    onClick={onPlayButtonClick}
                    height="30px"
                    width="30px"
                    alt="Play"
                />
                <img
                    src={PauseBackground}
                    onClick={onPauseButtonClick}
                    height="30px"
                    width="30px"
                    alt="Pause"
                />
                <button onClick={onNextMoveClick}>
                    <span>→</span>
                    <br />
                    <span>Next Move</span>
                </button>
            </div>
            <div>
                {[1, 2, 5, 10, 50, 100, 250].map(speed => (
                    <button key={speed} onClick={onAutoPlaySpeedClickFactory(speed)}>
                        {speed}x
                    </button>
                ))}
            </div>
        </div>
    );
};

export default PlayBar;
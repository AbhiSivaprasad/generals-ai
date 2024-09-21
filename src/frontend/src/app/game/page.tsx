'use client';

import React, { useEffect, useRef, useState } from 'react';
import Board from '../../components/board';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import socket, { BoardStateMessage, BoardUpdateMessage } from '../../socketio/client';
import { coalesceMoveQueues, patch } from '../../utils/board';
import { Action, CellState, serverActionToAction } from '../types/globals';
import Modal from '@/components/Modal';
import Button from '@/components/Button';

interface GamePlayingState {
    isOver: false
}

interface GameOverState {
    isOver: true,
    userIsWinner: boolean
}

type GameState = GamePlayingState | GameOverState

const GamePage: React.FC = () => {
    const searchParams = useSearchParams();  // Extract the dynamic route parameter
    const router = useRouter();
    const [moveQueue, setMoveQueue] = useState<Action[]>([]);
    const [landCounts, setLandCounts] = useState<number[]>([]);
    const [armyCounts, setArmyCounts] = useState<number[]>([]);
    const [playerNames, setPlayerNames] = useState<string[]>([]);
    const [board, setBoard] = useState<CellState[][] | null>(null);
    // The index of the player in the game
    const [playerIndex, setPlayerIndex] = useState<number | null>(null);
    const [fogOfWarEnabled, setFogOfWarEnabled] = useState(true);
    const [gameState, setGameState] = useState<GameState>({ isOver: false })
    const [boardPosition, setBoardPosition] = useState({ offsetX: 200, offsetY: 200, zoom: 1 });
    const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);

    // The server will send back the number of moves it has consumed,
    // but if the user has already reset the queue these moves should not be consumed.
    const hasQueueBeenResetSinceLastServerUpdate = useRef(false);

    const selectedCellRef = useRef<[number, number] | null>(null);
    useEffect(() => {
        selectedCellRef.current = selectedCell;
    }, [selectedCell]);

    // Add a listener for the move queue
    const keyPressHandlerFactory = (boardRows: number, boardCols: number) => (event: KeyboardEvent) => {
        if (event.key === 'w' || event.key === "ArrowUp") {
            if (selectedCellRef.current) {
                const [rowIndex, columnIndex] = selectedCellRef.current;
                setMoveQueue((currentMoveQueue) => {
                    const newMoveQueue: Action[] = [...currentMoveQueue, {
                        rowIndex,
                        columnIndex,
                        direction: 'up'
                    }];
                    socket.emit('set_move_queue', newMoveQueue);
                    return newMoveQueue;
                });
                setSelectedCell([Math.max(0, rowIndex - 1), columnIndex])
            }
        }
        if (event.key === 's' || event.key === 'ArrowDown') {
            if (selectedCellRef.current) {
                const [rowIndex, columnIndex] = selectedCellRef.current;
                setMoveQueue((currentMoveQueue) => {
                    const newMoveQueue: Action[] = [...currentMoveQueue, {
                        rowIndex,
                        columnIndex,
                        direction: 'down'
                    }];
                    socket.emit('set_move_queue', newMoveQueue);
                    return newMoveQueue;
                });
                setSelectedCell([Math.min(rowIndex + 1, boardRows - 1), columnIndex])
            }
        }
        if (event.key === 'a' || event.key === 'ArrowLeft') {
            if (selectedCellRef.current) {
                const [rowIndex, columnIndex] = selectedCellRef.current;
                setMoveQueue((currentMoveQueue) => {
                    const newMoveQueue: Action[] = [...currentMoveQueue, {
                        rowIndex,
                        columnIndex,
                        direction: 'left'
                    }];
                    socket.emit('set_move_queue', newMoveQueue);
                    return newMoveQueue;
                });
                setSelectedCell([rowIndex, Math.max(0, columnIndex - 1)])
            }
        }
        if (event.key === 'd' || event.key === 'ArrowRight') {
            if (selectedCellRef.current) {
                const [rowIndex, columnIndex] = selectedCellRef.current;
                setMoveQueue((currentMoveQueue) => {
                    const newMoveQueue: Action[] = [...currentMoveQueue, {
                        rowIndex,
                        columnIndex,
                        direction: 'right'
                    }];
                    socket.emit('set_move_queue', newMoveQueue);
                    return newMoveQueue;
                });
                setSelectedCell([rowIndex, Math.min(columnIndex + 1, boardCols - 1)])
            }
        }
        if (event.key === 'q') {
            hasQueueBeenResetSinceLastServerUpdate.current = true;
            setSelectedCell(null);
            setMoveQueue([]);
            socket.emit('set_move_queue', []);
        }
        if (event.key === 'Escape' || event.key === ' ') {
            setSelectedCell(null);
        }
        if (event.key === 'e') {
            setMoveQueue((currentMoveQueue) => {
                const lastMove = currentMoveQueue[currentMoveQueue.length - 1];
                console.log("last move", lastMove);
                if (!lastMove) {
                    console.log("no last move")
                    return currentMoveQueue;
                }
                setSelectedCell([lastMove.rowIndex, lastMove.columnIndex]);
                const newMoveQueue = currentMoveQueue.slice(0, -1);
                socket.emit('set_move_queue', newMoveQueue);
                return newMoveQueue;
            });
        }
    }

    useEffect(() => {
        // Support google-maps-style zooming in and out, where the mouse
        // position determines a fixed point on the board
        const handleWheel = (event: WheelEvent) => {
            setBoardPosition(({ zoom: previousZoom, offsetX: previousOffsetX, offsetY: previousOffsetY }) => {
                // Calculate the new zoom such that scroll up zooms in and
                // scroll down zooms out
                const newZoom = Math.max(0.5, Math.min(2, previousZoom * (event.deltaY / 2000 + 1)))
                // Now, we calculate the new offsets such that the mouse's
                // coordinates end up in the same position in client space
                const fixedPointWorldCoords = { x: (event.clientX - previousOffsetX) / previousZoom, y: (event.clientY - previousOffsetY) / previousZoom }
                // If the coords are 0, 0, and we zoom from 1 to 1.1, a point x,
                // y would move to x * 1.1, y * 1.1

                // Let's calculate what would happen to our fixed point if we
                // just applied the zoom
                const newFixedPointClientCoords = { x: fixedPointWorldCoords.x * newZoom + previousOffsetX, y: fixedPointWorldCoords.y * newZoom + previousOffsetY }

                // Now, we need to change the offset to back out this difference
                const offsetDelta = { x: event.clientX - newFixedPointClientCoords.x, y: event.clientY - newFixedPointClientCoords.y }

                return { zoom: newZoom, offsetX: previousOffsetX + offsetDelta.x, offsetY: previousOffsetY + offsetDelta.y }
            });
        };
        window.addEventListener('wheel', handleWheel);
        return () => window.removeEventListener('wheel', handleWheel);
    }, []);

    useEffect(() => {
        const opponentType = searchParams.get('opponentType') as "human" | "random";
        socket.emit('join-game', { opponentType });
        socket.on('game_start', (board_state: BoardStateMessage) => {
            // Save the board state to local storage
            setBoard(board_state.board_state);
            setPlayerIndex(board_state.player_index)
            setPlayerNames(board_state.player_names)
            console.log("adding keypress listener")
            window.addEventListener('keypress', keyPressHandlerFactory(board_state.board_state.length, board_state.board_state[0].length));
        })

        socket.on('game_over', (over_message) => {
            console.log("game over!", over_message)
            if (over_message.reason == "player_disconnected") {
                setGameState({ isOver: true, userIsWinner: true })
            } else if (over_message.reason === "player_won") {
                setGameState({ isOver: true, userIsWinner: true })
            }
        })

        socket.on('board_update', (board_update: BoardUpdateMessage) => {
            hasQueueBeenResetSinceLastServerUpdate.current = false;
            setBoard((currentBoard) => {
                if (!currentBoard) {
                    return currentBoard;
                }
                const newBoard = structuredClone(currentBoard);
                patch(newBoard, board_update.board_diff);
                return newBoard;
            });
            console.log('player scores are', board_update.player_scores)
            setLandCounts(board_update.land_counts);
            setArmyCounts(board_update.army_counts);
            // TODO: Unify the client and server move queue types
            setMoveQueue((moveQueue) => {
                return hasQueueBeenResetSinceLastServerUpdate.current ? moveQueue : moveQueue.slice(board_update.server_consumed_moves);
            });
        });


    }, []);

    // useEffect(() => {
    //     console.log("board", board);
    // }, [board])

    if (!board) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
                <h1 className="text-4xl font-bold mb-4 text-gray-800">Waiting for game</h1>
                <p className="text-xl text-gray-600">
                    Please wait while we find a match for you
                </p>
                <div className="mt-8">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500"></div>
                </div>
            </div>
        );
    }




    return (
        <div className="overflow-hidden">
            <div className="text-2xl font-bold text-center py-4 bg-blue-600 text-white">Playing Game</div>
            <div className='fixed top-4 right-4 w-64 bg-white rounded-lg shadow-lg p-4'>
                <h2 className="text-xl font-semibold mb-2">Scores</h2>
                <table className="w-full">
                    <thead>
                        <tr className="bg-gray-100">
                            <th className="py-2">Player</th>
                            <th className="py-2">Land</th>
                            <th className="py-2">Army</th>
                        </tr>
                    </thead>
                    <tbody>
                        {playerNames.map((name, index) => (
                            <tr key={index} className="border-b">
                                <td className="py-2">{name}</td>
                                <td className="py-2 text-center">{landCounts[index]}</td>
                                <td className="py-2 text-center">{armyCounts[index]}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="fixed bottom-4 left-4 bg-white rounded-lg shadow-lg p-4 flex items-center space-x-2">
                <span className="font-medium">Fog of War</span>
                <label className="inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={fogOfWarEnabled} onChange={(e) => setFogOfWarEnabled(e.target.checked)} className="sr-only peer" />
                    <div className="relative w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                </label>
            </div>
            <div>
                <Button onClick={() => router.push('/')}>Back to Main Menu</Button>
            </div>
            {gameState.isOver && <Modal><div className='flex flex-col space-y-4 items-center'><div>Game over</div><div>{"You " + (gameState.userIsWinner ? "won!" : " lost :(")}</div><Button onClick={() => window.location.reload()}>Play Again</Button></div></Modal>}

            {board && playerIndex !== null && <div className='absolute' style={{ left: boardPosition.offsetX, top: boardPosition.offsetY }}>
                <Board fogOfWarEnabled={fogOfWarEnabled} zoom={boardPosition.zoom} data={board} selectedCell={selectedCell} onCellSelect={(rowIndex, columnIndex) => setSelectedCell([rowIndex, columnIndex])} playerIndex={playerIndex} moveQueue={moveQueue} />
            </div>}
            {!board && <p>Loading board...</p>}
        </div>
    );
};

export default GamePage;
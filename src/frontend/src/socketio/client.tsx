import { io, Socket } from "socket.io-client";
import { Action, BoardDiff, BoardState } from "../app/types/globals";

export interface BoardStateMessage {
    board_state: BoardState;
    player_index: number;
    game_id: string;
}

export interface BoardUpdateMessage {
    board_diff: BoardDiff;
    move_queue: ServerAction[];
}

// TODO: Should this type be less hacky?
export interface GameOverMessage {
    reason: "player_disconnected" | "player_won"
    player_id: string
}

export interface JoinGameMessage {
    playBot: boolean
}

interface ServerToClientEvents {
    // Define the types of events received from the server
    connect: () => void;
    disconnect: () => void;
    message: (message: string) => void;
    game_start: (board_state: BoardStateMessage) => void;
    board_update: (board_updates: BoardUpdateMessage) => void;
    game_over: (disconnected: GameOverMessage) => void;
}

interface ClientToServerEvents {
    // Define the types of events sent to the server
    message: (message: string) => void;
    "join-game": (message: JoinGameMessage) => void;
    set_move_queue: (move_queue: Action[]) => void;
    set_username: (username: string) => void;
}

// Create a socket instance
const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io("http://localhost:8000");

// Handle connection event
socket.on("connect", () => {
    console.log("Connected to server");
});

// Handle disconnection event
socket.on("disconnect", () => {
    console.log("Disconnected from server");
});

// Handle custom event from server
socket.on("message", (message: string) => {
    console.log("Received message from server:", message);
});

// Send a message to the server
socket.emit("message", "Hello from the client!");

export default socket;
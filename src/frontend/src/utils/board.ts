import {
  Action,
  BoardDiff,
  BoardState,
  ServerAction,
} from "../app/types/globals";

/**
 * apply a move to a board state and return new board state
 * @param board
 * @param diffs: list of diff
 * @returns {*}
 */
export const patch = (board: BoardState, diffs: BoardDiff) => {
  for (let i = 0; i < diffs.length; i++) {
    const diff = diffs[i];
    let updateCell = board[diff.y][diff.x];

    // TODO: This is pretty ripe for bugs...can we just set the whole thing?
    updateCell.army = diff.army;
    updateCell.type = diff.type;
    updateCell.player_index = diff.player_index;
    updateCell.player_visibilities = diff.player_visibilities;
  }
};

export const isSameMove = (move1: Action, move2: Action) => {
  return (
    move1.columnIndex === move2.columnIndex &&
    move1.rowIndex === move2.rowIndex &&
    move1.direction === move2.direction
  );
};

// Coalesces using some pretty specific logic.
// If the movequeue is simply an extension of the new move queue, then we return
// the original move queue.
// If the new move queue is empty, we return empty.
// Otherwise, we assume that the new move queue consumed some actions from the
// beginning, and the original move queue might have some additional actions
// from the client at the end, so we return the original move queue - the piece
// of the move queue that we believe was consumed by the server.
export const coalesceMoveQueues = (
  moveQueue: Action[],
  newMoveQueue: Action[]
) => {
  if (newMoveQueue.length === 0) {
    return [];
  }

  let consumedIndex = 0;
  while (
    consumedIndex < newMoveQueue.length &&
    !isSameMove(moveQueue[consumedIndex], newMoveQueue[0])
  ) {
    consumedIndex++;
  }

  return moveQueue.slice(consumedIndex);
};

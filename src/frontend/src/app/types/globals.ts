export type CellState = {
  x: number;
  y: number;
  type: "normal" | "general" | "city" | "mountain";
  army: number;
  player_index: null | 0 | 1;
  player_visibilities: [boolean, boolean];
};

export type ServerAction = {
  rowIndex: number;
  columnIndex: number;
  doNothing: boolean;
  direction: 0 | 1 | 2 | 3;
};

export type Action = {
  rowIndex: number;
  columnIndex: number;
  direction: "up" | "down" | "left" | "right";
};

const directionMap = {
  0: "up",
  1: "down",
  2: "left",
  3: "right",
} as const;

export const serverActionToAction = (serverAction: ServerAction): Action => {
  return {
    rowIndex: serverAction.rowIndex,
    columnIndex: serverAction.columnIndex,
    direction: directionMap[serverAction.direction],
  };
};

export type BoardState = CellState[][]; // y = row, x = column

export type BoardDiff = CellState[]; // a list of cells that have been updated

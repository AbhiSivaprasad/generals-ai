import Cell from "../cell";
export type CellState = {
  x: number;
  y: number;
  type: "normal" | "general" | "city" | "mountain";
  army: number;
  player_index: null | 0 | 1;
  player_visibilities: [boolean, boolean];
};

export type BoardState = CellState[][]; // y = row, x = column

export type BoardDiff = CellState[]; // a list of cells that have been updated

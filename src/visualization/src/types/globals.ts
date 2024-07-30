export type CellState = { 
    "x": number, 
    "y": number, 
    "type": "normal" | "general" | "city" | "mountain",
    "army": number, 
    "player_index": null | 0 | 1,
    "player_visibilities": [boolean, boolean],
};

export type BoardState = CellState[][] // y = row, x = column

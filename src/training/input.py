import torch
from src.environment.board import Board
from src.environment.tile import Tile, TileType


def convert_state_to_tensor(state: Board, n_players: int, fog_of_war: bool):
    return torch.stack(
        [
            convert_state_to_tensor_for_player(state, player_index, fog_of_war)
            for player_index in range(n_players)
        ]
    )


def convert_state_to_tensor_for_player(
    state: Board, player_index: int, fog_of_war: bool
):
    state_tensor = torch.stack(
        [
            convert_tile_to_tensor(tile, player_index, fog_of_war)
            for row in enumerate(state.grid)
            for tile in enumerate(row)
        ]
    ).reshape(-1, state.num_rows, state.num_cols)
    return state_tensor


def convert_tile_to_tensor(tile: Tile, player_index: int, fog_of_war: bool):
    tile_state = torch.zeros(get_input_channel_dimension_size(fog_of_war))
    has_vision = tile.player_visibilities[player_index] or not fog_of_war

    # indices 0, 1 represents player 0, 1
    if tile.player_index is not None:
        tile_state[tile.player_index] = 1

    # index 2 represents the number of troops on the tile
    tile_state[2] = tile.army if tile.army is not None else 0

    if tile.type == TileType.NORMAL and has_vision:
        tile_type = 0
    elif tile.type == TileType.MOUNTAIN and has_vision:
        tile_type = 1
    elif tile.type == TileType.GENERAL and has_vision:
        tile_type = 2
    elif tile.type == TileType.CITY and has_vision:
        tile_type = 3
    elif (
        tile.type == TileType.CITY or tile.type == TileType.MOUNTAIN
    ) and not has_vision:
        tile_type = 4
    elif (
        tile.type == TileType.GENERAL or tile.type == TileType.NORMAL
    ) and not has_vision:
        tile_type = 5

    tile_state[tile_type + 3] = 1
    return tile_state


def get_input_channel_dimension_size(fog_of_war: bool):
    return 9 if fog_of_war else 7

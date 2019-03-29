'''
	@ Harris Christiansen (code@HarrisChristiansen.com)
	Generals.io Automated Client - https://github.com/harrischristiansen/generals-bot
	Tile: Objects for representing Generals IO Tiles
'''

from queue import Queue
from .constants import *


class Tile(object):
    def __init__(self, board, x, y):
        # Public Properties
        self.x = x                # Integer X Coordinate
        self.y = y                # Integer Y Coordinate
        self.type = TILE_FOG      # Integer Tile Type (TILE_OBSTACLE, TILE_FOG, TILE_MOUNTAIN, TILE_EMPTY, or player_ID)
        self.turn_captured = 0    # Integer Turn Tile Last Captured
        self.turn_held = 0        # Integer Last Turn Held
        self.army = 0             # Integer Army Count
        self.is_city = False      # Boolean is_city
        self.is_general = False   # Boolean is_general

        # Private Properties
        self._board = board       # Pointer to Board Object
        self._general_index = -1  # Player Index if tile is a general

    def __repr__(self):
        return "(%d,%d) %d (%d)" % (self.x, self.y, self.type, self.army)

    def __lt__(self, other):
        return self.army < other.army

    def serialize(self):
        return {
            'x': self.x,
            'y': self.y,
            'type': self.type,
            'army': self.army,
            'isCity': self.is_city,
            'isGeneral': self.is_general
        }

    def set_neighbors(self, board):
        self._board = board
        self._set_neighbors()

    def update(self, board, new_tile_type, army, is_city=False, is_general=False):
        """
        given parameters for new tile, update state and parent state
        """
        self._board = board

        if (self.type < 0 or new_tile_type >= TILE_MOUNTAIN
            or (new_tile_type < TILE_MOUNTAIN and self.is_self())):  # Tile should be updated
            if (new_tile_type >= 0 or self.type >= 0) and self.type != new_tile_type:  # Remember Discovered Tiles
                self.turn_captured = board.turn
                if self.type >= 0:
                    board.tiles[self.type].remove(self)
                if new_tile_type >= 0:
                    board.tiles[new_tile_type].append(self)
            if new_tile_type == board.player_index:
                self.turn_held = board.turn
            self.type = new_tile_type
        if self.army == 0 or army > 0 or new_tile_type >= TILE_MOUNTAIN:  # Remember Discovered Armies
            self.army = army

        if is_city:
            self.is_city = True
            if self not in board.cities:
                board.cities.append(self)

            assert(not is_general)
            assert(self._general_index == -1)
            # mark as not general unclear why necessary
            # self.is_general = False
            # if self._general_index != -1 and self._general_index < 8:
            #     board.generals[self._general_index] = None
            #     self._general_index = -1
        elif is_general:
            self.is_general = True
            board.generals[new_tile_type] = self
            self._general_index = self.type

    ################################ Tile Properties ################################

    def is_self(self):
        return (self._board.player_index is None            # Game master's board--does not belong to player
                or self.type == self._board.player_index)   # Player's board

    def distance_to(self, dest):
        if dest is not None:
            return abs(self.x - dest.x) + abs(self.y - dest.y)
        return 0

    def copy(self, tile):
        if tile.is_city and not self.is_city:
            self._board.cities.append(self)
        elif not tile.is_city and self.is_city:
            self._board.cities.remove(self)

        if tile.is_general and not self.is_general:
            self._board.generals[tile.type] = self  # if tile is general then type must be player index

        self.type = tile.type
        self.army = tile.army
        self.is_city = tile.is_city
        self.is_general = tile.is_general
        self.turn_captured = tile.turn_captured
        self.turn_held = tile.turn_held

    def neighbors(self, include_cities=True):
        neighbors = []
        for tile in self._neighbors:
            if (tile.type != TILE_OBSTACLE or tile.is_city or tile.is_general) \
                    and tile.type != TILE_MOUNTAIN \
                    and (include_cities or not tile.is_city):
                neighbors.append(tile)
        return neighbors

    def isValidTarget(self):  # Check tile to verify reachability
        if self.type < TILE_EMPTY:
            return False
        for tile in self.neighbors():
            if tile.turn_held > 0:
                return True
        return False

    def is_on_team(self):
        return self.is_self()

    def shouldNotAttack(self):
        if self.is_on_team():
            return True
        if self.type in self._board.do_not_attack_players:
            return True
        return False

    ################################ Select Other Tiles ################################

    def nearest_target_tile(self):
        max_target_army = self.army * 2 + 14

        dest = None
        dest_distance = 9999
        for x in range(self._board.cols):  # Check Each Square
            for y in range(self._board.rows):
                tile = self._board.grid[y][x]
                if not tile.isValidTarget() or tile.shouldNotAttack() or tile.army > max_target_army:  # Non Target Tiles
                    continue

                distance = self.distance_to(tile)
                if tile.is_general:  # Generals appear closer
                    distance = distance * 0.09
                elif tile.is_city:  # Cities vary distance based on size, but appear closer
                    distance = distance * sorted((0.17, (tile.army / (3.2 * self.army)), 20))[1]

                if tile.tile == TILE_EMPTY:  # Empties appear further away
                    distance = distance * 4.3
                if tile.army > self.army:  # Larger targets appear further away
                    distance = distance * (1.5 * tile.army / self.army)
                if distance < dest_distance:
                    dest = tile
                    dest_distance = distance

        return dest

    ################################ Pathfinding ################################

    def path_to(self, dest, include_cities=False):
        if dest == None:
            return []

        frontier = Queue()
        frontier.put(self)
        came_from = {}
        came_from[self] = None
        army_count = {}
        army_count[self] = self.army

        while not frontier.empty():
            current = frontier.get()

            if current == dest:  # Found Destination
                break

            for next in current.neighbors(include_cities=include_cities):
                if next not in came_from and (next.is_on_team() or next == dest or next.army < army_count[current]):
                    # priority = self.distance(next, dest)
                    frontier.put(next)
                    came_from[next] = current
                    if next.is_on_team():
                        army_count[next] = army_count[current] + (next.army - 1)
                    else:
                        army_count[next] = army_count[current] - (next.army + 1)

        # if game master is caller
        if self._board.player_index is None:
            if dest not in came_from:
                return False
            return True

        # if player is caller
        if dest not in came_from:  # Did not find dest
            if include_cities:
                return []
            else:
                return self.path_to(dest, include_cities=True)

        # Create Path List
        path = _path_reconstruct(came_from, dest)

        return path

    ################################ PRIVATE FUNCTIONS ################################

    def _set_neighbors(self):
        x = self.x
        y = self.y

        neighbors = []
        for dy, dx in DIRECTIONS:
            if self._board.is_valid_position(x + dx, y + dy):
                tile = self._board.grid[y + dy][x + dx]
                neighbors.append(tile)

        self._neighbors = neighbors
        return neighbors


def _path_reconstruct(came_from, dest):
    current = dest
    path = [current]
    try:
        while came_from[current] != None:
            current = came_from[current]
            path.append(current)
    except KeyError:
        pass
    path.reverse()

    return path

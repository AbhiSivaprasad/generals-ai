import logging
import json
import threading
import time
from websocket import create_connection, WebSocketConnectionClosedException

from src.move import Move
from src.graphics.tile import Tile
from src.graphics.board import Board
from src.graphics.constants import *

_ENDPOINT = "ws://botws.generals.io/socket.io/?EIO=3&transport=websocket"
_REPLAY_URL = "http://bot.generals.io/replays/"

_RESULTS = {
  "game_update": "",
  "game_won": "win",
  "game_lost": "lose",
}

class Generals(object):
  def __init__(self, userid, username=None, mode="1v1", gameid=None,
         force_start=True, region=None, col=19, row=19):

    self.userid = userid
    self.username = username
    self.mode = mode
    self.gameid = gameid
    self.force_start = force_start

    logging.debug("Creating connection")
    self._ws = create_connection(_ENDPOINT)
    self._lock = threading.RLock()

    logging.debug("Starting heartbeat thread")
    _spawn(self._start_sending_heartbeat)

    logging.debug("Joining game")

    if isinstance(username, str):
      self._send(["set_username", userid, username])


    # join a game
    self._join_game()

    # set up the board
    self.board = Board(rows=row, cols=col, player_index=None)
    self.grid = [ # 2D List of Tile Objects
       [Tile(self.board, x, y) for x in range(col)]
       for y in range(row)
      ]

    self.board.set_grid(self.grid)

    self._seen_update = False
    self._move_id = 1
    self._start_data = {}
    self._stars = []
    self._map = []
    self._cities = []

  def _join_game(self):
    if self.mode == "private":
      if self.gameid is None:
        raise ValueError("Gameid must be provided for private games")
      self._send(["join_private", self.gameid, self.userid])

    elif self.mode == "1v1":
      self._send(["join_1v1", self.userid])

    elif self.mode == "team":
      if self.gameid is None:
        raise ValueError("Gameid must be provided for team games")
      self._send(["join_team", self.f, self.userid])

    elif self.mode == "ffa":
      self._send(["play", self.userid])

    else:
      raise ValueError("Invalid mode")

    # delay so that the force start goes through
    time.sleep(2)
    self._send(["set_force_start", self.gameid, self.force_start])

  def move(self, x1, y1, x2, y2, move_half=False):
    if not self._seen_update:
      raise ValueError("Cannot move before first map seen")

    cols = self._map[0]
    a = y1 * cols + x1
    b = y2 * cols + x2
    self._send(["attack", a, b, move_half, self._move_id])
    self._move_id += 1

  def get_updates(self):
    while True:
      try:
        msg = self._ws.recv()
      except WebSocketConnectionClosedException:
        break

      if not msg.strip():
        break

      # ignore heartbeats and connection acks
      if msg in {"3", "40"}:
        continue

      # print("msg:", msg)

      # remove numeric prefix
      while msg and msg[0].isdigit():
        msg = msg[1:]

      msg = json.loads(msg)
      if not isinstance(msg, list):
        continue

      if msg[0] == "error_user_id":
        raise ValueError("Already in game")
      elif msg[0]== 'pre_game_start':
        logging.info("Game Prepare to Start")
      elif msg[0] == "game_start":
        logging.info("Game info: {}".format(msg[1]))
        self._start_data = msg[1]
      elif msg[0] == "game_update":
        yield self._make_update(msg[1])
      elif msg[0] in ["game_won", "game_lost"]:
        yield self._make_result(msg[0], msg[1])
        # break
      else:
        logging.info("Unknown message type: {}".format(msg))

  def close(self):
    self._ws.close()


  def _make_update(self, data):
    print("MAP DIFF", data['map_diff'])
    _apply_diff(self._map, data['map_diff'])
    _apply_diff(self._cities, data['cities_diff'])
    if 'stars' in data:
      self._stars[:] = data['stars']

    row, col = self._map[1], self._map[0]
    self._seen_update = True

    print(not (row == self.board.rows and col == self.board.col))

    # if the size of the board given isn't what's expected, quit and rejoin
    if not (row == self.board.rows and col == self.board.col):
      print('leaving game')
      self._send(["leave_game"])
      self._join_game()
      return {
        'complete': False,
        'bad_board': True
      }


    # self.board = Board(rows=row, cols=col, player_index=None)
    # self.grid = [ # 2D List of Tile Objects
    #    [Tile(self.board, x, y) for x in range(col)]
    #    for y in range(row)
    #   ]

    self.board.set_grid(self.grid)

    pi = self._start_data['playerIndex']
    generals = [(-1, -1) if g == -1 else (g // col, g % col)
             for g in data['generals']]

    gen_y, gen_x = generals[pi]

    cities = [(c // col, c % col) for c in self._cities]

    army_grid = [[self._map[2 + y*col + x]
             for x in range(col)]
             for y in range(row)]

    tile_grid = [[self._map[2 + col*row + y*col + x]
             for x in range(col)]
             for y in range(row)]

    for x in range(row):
      for y in range(col):
       tile = self.grid[x][y]

       tile.type = tile_grid[x][y]
       tile.army = army_grid[x][y]

       if x == gen_x and y == gen_y:
        tile.is_general = True
        self.board.generals[0] = tile

       if (x,y) in cities:
        tile.is_city = True
        self.board.cities.append(tile)

    # is list of all moves, not just legal ones
    for x in range(row):
      for y in range(col):
        tile = self.grid[x][y]

        if tile.type > -1:
          for dx, dy in DIRECTIONS:
            if self.board.is_valid_position(tile.x + dx, tile.y + dy):
              # the neighboring tile is not a mountain so we have found a valid move
              self.board.legal_moves.add(
                Move(startx=tile.x,
                     starty=tile.y,
                     destx=tile.x + dx,
                     desty=tile.y + dy)
              )


    return {
      'bad_board': False,
      'complete': False,
      'turn': data['turn'],
      'board': self.board
    }
    # sort by player index
    # scores = {d['i']: d for d in data['scores']}
    # scores = [scores[i] for i in range(len(scores))]
    # return {
    #   'complete': False,
    #   'row': row,
    #   'col': col,
    #   'player_index': self._start_data['playerIndex'],
    #   'turn': data['turn'],
    #   'army_grid': [[self._map[2 + y*col + x]
    #          for x in range(col)]
    #          for y in range(row)],
    #   'tile_grid': [[self._map[2 + col*row + y*col + x]
    #          for x in range(col)]
    #          for y in range(row)],
    #   'lands': [s['tiles'] for s in scores],
    #   'armies': [s['total'] for s in scores],
    #   'alives': [not s['dead'] for s in scores],
    #   'generals': [(-1, -1) if g == -1 else (g // col, g % col)
    #         for g in data['generals']],
    #   'cities': [(c // col, c % col) for c in self._cities],
    #   'usernames': self._start_data['usernames'],
    #   'teams': self._start_data.get('teams'),
    #   'stars': self._stars,
    #   'replay_url': _REPLAY_URL + self._start_data['replay_id'],
    # }

  def _make_result(self, update, data):
    return {
      'complete': True,
      'result': update == "game_won",
      'player_index': self._start_data['playerIndex'],
      'usernames': self._start_data['usernames'],
      'teams': self._start_data.get('teams'),
      'stars': self._stars,
      'replay_url': _REPLAY_URL + self._start_data['replay_id'],
    }

  def _start_sending_heartbeat(self):
    while True:
      try:
        with self._lock:
          self._ws.send("2")
      except WebSocketConnectionClosedException:
        break
      time.sleep(10)

  def _send(self, msg):
    try:
      with self._lock:
        self._ws.send("42" + json.dumps(msg))
    except WebSocketConnectionClosedException:
      pass


def _spawn(f):
  t = threading.Thread(target=f)
  t.daemon = True
  t.start()


def _apply_diff(cache, diff):
  i = 0
  a = 0
  while i < len(diff) - 1:

    # offset and length
    a += diff[i]
    n = diff[i+1]

    cache[a:a+n] = diff[i+2:i+2+n]
    a += n
    i += n + 2

  if i == len(diff) - 1:
    cache[:] = cache[:a+diff[i]]
    i += 1

  assert i == len(diff)
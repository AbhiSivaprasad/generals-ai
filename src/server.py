from pathlib import Path
from flask import Flask, jsonify, request
from flask import make_response

import os

__dirname__ = os.path.dirname(__file__)
ROOT_DIR = Path(__dirname__).parent

app = Flask(__name__)


@app.route("/replay/<replay_id>")
def serve_replay(replay_id):
    replay_file = f"{replay_id}.json"
    replay_path = ROOT_DIR / f"resources/replays/{replay_file}"

    with open(replay_path, "r") as file:
        r = make_response(file.read())
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r


if __name__ == "__main__":
    app.run(debug=True, port=8000)

# TASKS
# implement generals.io api
# implement socket server to visualize games in real time
# style visualization page

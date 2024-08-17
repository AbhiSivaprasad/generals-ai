from pathlib import Path
from flask import Flask, jsonify, request
from flask import make_response

import os

__dirname__ = os.path.dirname(__file__)
ROOT_DIR = Path(__dirname__).parent

app = Flask(__name__)


@app.route("/replay/<path:replay_path>")
def serve_replay(replay_path):
    print("request for replay path:", replay_path)

    # Construct the full path
    replay_full_path = ROOT_DIR / "resources/replays" / f"{replay_path}.json"

    # Check if the file exists and is within the allowed directory
    if not replay_full_path.is_file() or not replay_full_path.is_relative_to(
        ROOT_DIR / "resources/replays"
    ):
        return "Replay not found", 404

    with open(replay_full_path, "r") as file:
        r = make_response(file.read())
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r


if __name__ == "__main__":
    app.run(debug=True, port=8000)

# TASKS
# implement socket server to visualize games in real time
# style visualization page

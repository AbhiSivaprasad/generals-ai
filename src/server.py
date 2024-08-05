from flask import Flask, request
from flask import make_response

import argparse

import os

__dirname__ = os.path.dirname(__file__)

app = Flask(__name__)

@app.route('/replay')
def replay():
    with open(os.path.join(os.getcwd(), app.config['replay']), 'r') as file:
        r = make_response(file.read())
        r.headers['Access-Control-Allow-Origin'] = "*"
        return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, default=os.path.join(__dirname__, "../resources/replays/test_replay.txt"))
    args = parser.parse_args()
    
    app.config['replay'] = args.replay
    app.run(debug=True, port=8000)

# TASKS
# implement generals.io api
# implement socket server to visualize games in real time
# style visualization page

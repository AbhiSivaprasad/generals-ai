from flask import Flask, request
from flask import make_response

app = Flask(__name__)

@app.route('/replay')
def replay():
    with open('../resources/replays/{}.txt'.format(6300), 'r') as file:
        r = make_response(file.read())
        r.headers['Access-Control-Allow-Origin'] = "*"
        return r

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# TASKS
# implement generals.io api
# implement socket server to visualize games in real time
# style visualization page

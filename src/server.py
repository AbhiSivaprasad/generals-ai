from flask import Flask, request
app = Flask(__name__)

@app.route('/replay')
def replay():
    with open('../resources/replays/{}.txt'.format(6300), 'r') as file:
        return file.read()

if __name__ == '__main__':
    app.run()

# TASKS
# implement generals.io api
# implement socket server to visualize games in real time
# style visualization page

from flask import Flask
app = Flask(__name__)

@app.route('/replay')
def replay():
    with open('../resources/replays/temp.txt', 'r') as file:
        return file.read()

if __name__ == '__main__':
    app.run()

# TASKS
# implement generals.io api
# implement socket server to visualize games in real time
# style visualization page
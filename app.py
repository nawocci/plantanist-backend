from flask import Flask

app = Flask(__name__)

# placeholder route to / for testing which return hello world
@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

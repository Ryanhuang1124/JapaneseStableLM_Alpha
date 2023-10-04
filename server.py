from flask import Flask, Response, render_template
import time

app = Flask(__name__)

def stream_text():
    for i in range(1, 11):
        yield f"data: New text {i}\n\n"
        time.sleep(1)

@app.route('/')
def index():
    return render_template('server.html')

@app.route('/stream')
def stream():
    return Response(stream_text(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)

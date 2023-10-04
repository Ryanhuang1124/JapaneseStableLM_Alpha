from flask import Flask, render_template, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


dialogue = []

app = Flask(__name__)
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
streamer = TextIteratorStreamer(tok)
# Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.

def generate_response(input):
    inputs = tok([input], return_tensors="pt")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            generate_response(user_input)
            dialogue.append(("User", user_input))
            generated_text = ""
            dialogue.append(("AI", generated_text))
            for new_text in streamer:
                print(streamer)
                if new_text:
                    generated_text += new_text
                    dialogue.pop()
                    dialogue.append(("AI", generated_text))
                    render_template("index.html", dialogue=dialogue)
    return render_template("index.html", dialogue=dialogue)


if __name__ == "__main__":
    app.run(use_reloader=False)

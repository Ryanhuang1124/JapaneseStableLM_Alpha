#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, stream_with_context, request, Response
from flask import redirect, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
streamer = TextIteratorStreamer(tokenizer)
input = ""
dialogue = []


@app.route('/', methods=["GET", "POST"])
def index():
    global input
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            input = user_input.strip()
            return redirect('/stream')
    return render_template("app_stream_test.html", dialogue = dialogue)

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv

def generate(prompt):
    response = ""
    dialogue.append(("User", prompt))
    inputs = tokenizer([prompt], return_tensors="pt")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=128)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for item in streamer:
        response += item
        yield item
    dialogue.append(("AI", response))

@app.route('/stream')
def stream_view():
    rows = generate(input)
    return Response(stream_with_context(stream_template('app_stream_test.html', rows=rows, page_stream = True, input = input, dialogue = dialogue)))

if __name__ == '__main__':
    app.debug = False
    app.run()
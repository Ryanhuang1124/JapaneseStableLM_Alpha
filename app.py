from flask import Flask, render_template, request, redirect, stream_with_context, Response
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
from threading import Thread

app = Flask(__name__)

tokenizer = LlamaTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
)

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-base-alpha-7b",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.half()
model.eval()

streamer = TextIteratorStreamer(tokenizer)

input = ""
dialogue = []


if torch.cuda.is_available():
    model = model.to("cuda")



def generate_response(prompt):
    response = ""
    dialogue.append(("User", prompt))
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    
    generation_kwargs = dict(
    input_ids=input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=1,
    top_p=0.95,
    do_sample=True,
    streamer=streamer,)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for item in streamer:
        response += item
        yield item
    dialogue.append(("AI", response))


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


@app.route('/stream')
def stream_view():
    rows = generate_response(input)
    return Response(stream_with_context(stream_template('app_stream_test.html', rows=rows, page_stream = True, input = input, dialogue = dialogue)))


if __name__ == '__main__':
    app.debug = False
    app.run()

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
streamer = TextStreamer(tok)

# Despite returning the usual output, the streamer will also print the generated text to stdout.
print(type(model.generate(**inputs, streamer=streamer, max_new_tokens=20)))
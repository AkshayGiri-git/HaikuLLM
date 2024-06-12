import tiktoken
text = ''
def getData():
    with open('all_haiku.txt', 'r') as f:
        text = f.read()
    print(text[:100])
    return text

def tokenize():
    enc = tiktoken.encoding_for_model("gpt-4o")
    print(enc.encode(text))
getData()
tokenize()

# gptzip
#### Losslessly encode text natively with arithmetic coding and HuggingFace Transformers

Did you know that every time you download a language model to your computer, you're downloading powerful compression technology as well?

`gptzip` is a python library that uses pre-trained language models as string compressors. It's compatible out-of-the-box with language models from HuggingFace transformers and uses arithmetic coding (which is theoretically optimal) to compress strings based on language model probability distributions. 

This all works because of [Shannon's source coding theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem) which connects probability distributions and compression. Since language models like GPT-3 give us probabilities over strings, we can literally use them as compressors. gptzip makes this trivial.

### Installation
```pip install gptzip```

### Encoding

You can use gptzip to check the number of bytes a language model requires to encode a string (to compare against e.g. gzip or the original byte count):

```python
model = "gpt2"
lm = transformers.AutoModelForCausalLM.from_pretrained(model)
tokenizer = transformers.AutoTokenizer.from_pretrained(model)
string = "Sailing on the seven seas"
coder = gptzip.ArithmeticCoder(lm=lm, tokenizer=tokenizer)
code, num_padded_bits = coder.encode(
    string, 
    return_num_padded_bits=True, 
)
assert len(code) == 5
```

### Lossless encoding-and-decoding

Perhaps even more useful is to use gptzip as a true file compressor. In this case, `code`
```python
model = "gpt2"
lm = transformers.AutoModelForCausalLM.from_pretrained(model)
tokenizer = transformers.AutoTokenizer.from_pretrained(model)
string = "How much would could a woodchuck chuck?"
coder = gptzip.ArithmeticCoder(lm=lm, tokenizer=tokenizer)
code, num_padded_bits = coder.encode(
    string, 
    return_num_padded_bits=True, 
)
print(f"Code: {to_binary(code)} ({len(code)} bytes)")
decoded_string = coder.decode(code, num_padded_bits=num_padded_bits)
assert decoded_string == string
```


### Roadmap

Some features that would be nice to add:

- [ ] Other compression techniques such as Huffman
- [ ] Benchmarking against other compressions and add numbers to README
- [ ] Support for other language modeling softwares such as VLLM
- [ ] Compress multiple strings in batch

### Citation

Thanks to DeepMind implementation for helping me implement Arithmetic coding in Python. I learned a lot from their [implementation](https://github.com/google-deepmind/language_modeling_is_compression) and paper, [Language Modeling Is Compression](https://deepmind.google/research/publications/39768/). 

I also am indebted to Mark Nelson for his incredibly blog post [Data Compression With Arithmetic Coding](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html). It was invaluable for me while learning about this topic, especially the lossless implementation of arithmetic coding using binary fractions. It's one of the best blog posts that I have ever read.

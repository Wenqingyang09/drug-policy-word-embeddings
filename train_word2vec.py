import os
import re
from gensim.models import Word2Vec

# Load cleaned corpus
input_file = os.path.expanduser("~/Desktop/usa_corpus_clean.txt")

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize into sentences and words
text = text.lower()
text = re.sub(r"[^a-z\s]", " ", text)
sentences = [sentence.split() for sentence in text.split("\n") if len(sentence.strip()) > 10]

print(f"Total sentences: {len(sentences)}")

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=10,
    sg=1
)

# Save model
model_path = os.path.expanduser("~/Desktop/usa_word2vec.model")
model.save(model_path)
print("Model saved to:", model_path)

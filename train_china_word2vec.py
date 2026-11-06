import os
import re
import jieba
from gensim.models import Word2Vec

# Load cleaned Chinese corpus
input_file = os.path.expanduser("~/Desktop/china_corpus_clean.txt")

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize Chinese text using jieba
sentences = []
for line in text.split("\n"):
    line = line.strip()
    if len(line) < 5:
        continue
    words = list(jieba.cut(line))
    words = [w for w in words if len(w.strip()) > 0]
    if len(words) > 3:
        sentences.append(words)

print(f"Total sentences: {len(sentences)}")

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    sg=1
)

# Save model
model_path = os.path.expanduser("~/Desktop/china_word2vec.model")
model.save(model_path)
print("Model saved to:", model_path)

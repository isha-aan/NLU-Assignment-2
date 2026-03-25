from PyPDF2 import PdfReader

def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + " "
    
    return text


# Load your PDFs
reg_text = extract_pdf_text("academic_regulationsIITJ.pdf")
course_text = extract_pdf_text("CSE-Courses-Details.pdf")

print("PDF extraction done!")

import requests
from bs4 import BeautifulSoup

url = "https://www.iitj.ac.in/computer-science-engineering/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

web_text = soup.get_text()

print("Website extraction done!")

all_text = reg_text + " " + course_text + " " + web_text

print("Text combined!")

import re

def clean_text(text):
    # remove newlines
    text = re.sub(r'\n+', ' ', text)
    
    # remove numbers & special characters
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    
    # convert to lowercase
    text = text.lower()
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


cleaned_text = clean_text(all_text)

print("Text cleaned!")

# Tokenization
tokens = cleaned_text.split()

from gensim.parsing.preprocessing import STOPWORDS

tokens = [
    word for word in tokens 
    if word not in STOPWORDS and len(word) > 2
]

print("Tokenization + cleaning done!")

from collections import Counter

total_docs = 3
total_tokens = len(tokens)
vocab = set(tokens)
vocab_size = len(vocab)

print("Documents:", total_docs)
print("Total Tokens:", total_tokens)
print("Vocabulary Size:", vocab_size)

with open("clean_corpus.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens))

print("Corpus saved!")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white') \
            .generate(" ".join(tokens))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from gensim.models import Word2Vec

# Word2Vec needs list of sentences
# We'll treat entire corpus as one sentence (simple approach)

sentences = [tokens]

cbow_model = Word2Vec(
    sentences,
    vector_size=100,   # embedding dimension
    window=5,          # context window
    min_count=2,
    sg=0,              # 0 = CBOW
    negative=5
)

print("CBOW model trained!")

skipgram_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,              # 1 = Skip-gram
    negative=5
)

print("Skip-gram model trained!")
print("\nCBOW similar words to 'student':")
print(cbow_model.wv.most_similar("student", topn=5))

print("\nSkip-gram similar words to 'student':")
print(skipgram_model.wv.most_similar("student", topn=5))

# CBOW variations
cbow_100 = Word2Vec(sentences, vector_size=100, window=5, sg=0, negative=5)
cbow_200 = Word2Vec(sentences, vector_size=200, window=5, sg=0, negative=5)
cbow_w8 = Word2Vec(sentences, vector_size=100, window=8, sg=0, negative=5)
cbow_neg10 = Word2Vec(sentences, vector_size=100, window=5, sg=0, negative=10)

print("CBOW experiments done!")

# Skip-gram variations
sg_100 = Word2Vec(sentences, vector_size=100, window=5, sg=1, negative=5)
sg_200 = Word2Vec(sentences, vector_size=200, window=5, sg=1, negative=5)
sg_w8 = Word2Vec(sentences, vector_size=100, window=8, sg=1, negative=5)
sg_neg10 = Word2Vec(sentences, vector_size=100, window=5, sg=1, negative=10)

print("Skip-gram experiments done!")

words = ["research", "student", "phd", "exam"]

for word in words:
    print(f"\nCBOW - {word}:")
    if word in cbow_100.wv:
        print(cbow_100.wv.most_similar(word, topn=5))
    else:
        print("Word not found")

    print(f"Skip-gram - {word}:")
    if word in sg_100.wv:
        print(sg_100.wv.most_similar(word, topn=5))
    else:
        print("Word not found")


print("\n--- Analogy Experiments ---")

# 1
try:
    print("student : course :: professor : ?")
    print(sg_100.wv.most_similar(positive=["professor", "course"], negative=["student"], topn=3))
except:
    print("Analogy 1 failed")

# 2
try:
    print("\nbtech : ug :: mtech : ?")
    print(sg_100.wv.most_similar(positive=["mtech", "ug"], negative=["btech"], topn=3))
except:
    print("Analogy 2 failed")

# 3
try:
    print("\nlearning : data :: research : ?")
    print(sg_100.wv.most_similar(positive=["research", "data"], negative=["learning"], topn=3))
except:
    print("Analogy 3 failed")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# select some important words
words = list(cbow_100.wv.index_to_key)[:50]

# get vectors
cbow_vectors = [cbow_100.wv[word] for word in words]
sg_vectors = [sg_100.wv[word] for word in words]

# apply PCA
pca = PCA(n_components=2)

cbow_2d = pca.fit_transform(cbow_vectors)
sg_2d = pca.fit_transform(sg_vectors)

# plot CBOW
plt.figure(figsize=(8,6))
plt.scatter(cbow_2d[:,0], cbow_2d[:,1])

for i, word in enumerate(words):
    plt.annotate(word, (cbow_2d[i,0], cbow_2d[i,1]))

plt.title("CBOW Word Embeddings (PCA)")
plt.show()

# plot Skip-gram
plt.figure(figsize=(8,6))
plt.scatter(sg_2d[:,0], sg_2d[:,1])

for i, word in enumerate(words):
    plt.annotate(word, (sg_2d[i,0], sg_2d[i,1]))

plt.title("Skip-gram Word Embeddings (PCA)")
plt.show()
import math
import re

# Minimal stopwords list (expand if needed)
STOPWORDS = {
    'the','is','in','and','to','of','a','for','on','with','as','by','an','at',
    'from','that','this','it','be','are','was','were','or','has','have','had'
}

# Sentence tokenizer (basic)
def split_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

# Word tokenizer (basic)
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

# Term Frequency
def tf(word, sentence_tokens):
    return sentence_tokens.count(word) / len(sentence_tokens) if sentence_tokens else 0

# Inverse Document Frequency
def idf(word, sentences_tokens):
    count = sum(1 for sent in sentences_tokens if word in sent)
    return math.log(len(sentences_tokens) / (1 + count))

# TF-IDF Score
def tf_idf(sentence_tokens, sentences_tokens):
    score = 0
    for word in sentence_tokens:
        score += tf(word, sentence_tokens) * idf(word, sentences_tokens)
    return score

# Summarizer
def summarizer(text, length=3):
    sentences = split_sentences(text)
    sentences_tokens = [tokenize(sent) for sent in sentences]

    scores = {}
    for i, sent in enumerate(sentences):
        scores[sent] = tf_idf(sentences_tokens[i], sentences_tokens)

    # Rank sentences
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Pick top-k
    top_sentences = [s[0] for s in ranked[:length]]

    return ' '.join(top_sentences)

txt="""Born in Ranchi, Dhoni made his first class debut for Bihar in 1999. He made his debut for the Indian cricket team on 23 December 2004 in an ODI against Bangladesh and played his first test a year later against Sri Lanka. In 2007, he became the captain of the ODI side before taking over in all formats by 2008. Dhoni retired from test cricket in 2014, but continued playing in limited overs cricket till 2019. He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODIs."""
print(summarizer(txt))
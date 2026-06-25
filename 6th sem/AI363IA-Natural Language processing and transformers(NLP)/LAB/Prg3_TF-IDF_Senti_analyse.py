#nlp3
import numpy as np, math, re
from collections import Counter

STOP = {'a','an','the','and','or','but','if','while','with','of','at','by','for',
        'to','from','in','on','out','over','under','again','further','then','once',
        'here','there','all','any','both','each','few','more','most','other','some',
        'such','no','nor','not','only','own','same','so','than','too','very',
        'can','will','just','don','should','now','is','am','are','was','were',
        'be','been','being','have','has','had','do','does','did',
        'it','its','this','that','these','those',
        'i','my','me','we','our','you','your','he','she','they','them','their',
        'what','which','who','how','when','where','why'}

# ── Data ────────────────────────────────────────
import pandas as pd
df = pd.read_csv('Musical_instruments_reviews 4.csv').dropna(subset=['summary'])
X = df['summary'].values
y = np.where(df['overall'].values >= 4, 'positive', np.where(df['overall'].values <= 2, 'negative', 'neutral'))

np.random.seed(1); idx = np.random.permutation(len(X)); split = int(len(X) * 0.8)
tr, te = idx[:split], idx[split:]
X_tr, X_te, y_tr, y_te = [X[i] for i in tr], [X[i] for i in te], [y[i] for i in tr], [y[i] for i in te]

# ── Preprocess ──────────────────────────────────
def tokenize(text):
    return [w for w in re.sub(r'[^a-z0-9 ]', '', str(text).lower()).split() if w not in STOP]

X_tr_tok, X_te_tok = [tokenize(t) for t in X_tr], [tokenize(t) for t in X_te]

# ── Vocab + IDF (no smoothing) ──────────────────
freq = Counter(w for doc in X_tr_tok for w in doc)
vocab = list(freq); w2i = {w: i for i, w in enumerate(vocab)}; V = len(vocab); N = len(X_tr_tok)
idf = {w: math.log(N / sum(1 for doc in X_tr_tok if w in doc)) for w in vocab}

# ── TF-IDF ──────────────────────────────────────
def vectorize(docs):
    M = np.zeros((len(docs), V))
    for d, doc in enumerate(docs):
        tf = Counter(doc); total = len(doc) or 1
        for w, c in tf.items():
            if w in w2i: M[d, w2i[w]] = (c / total) * idf[w]
    return M

X_tv, X_ev = vectorize(X_tr_tok), vectorize(X_te_tok)

# ── Labels ──────────────────────────────────────
classes = sorted(set(y_tr)); c2i = {c: i for i, c in enumerate(classes)}; C = len(classes)
y_tv = np.array([c2i[c] for c in y_tr]); y_ev = np.array([c2i[c] for c in y_te])

# ── Train ───────────────────────────────────────
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

W = np.zeros((V, C)); b = np.zeros((1, C))
OH = np.eye(C)[y_tv]; np.random.seed(42)

for _ in range(50):
    idx = np.random.permutation(len(X_tv))
    for s in range(0, len(X_tv), 512):
        bi = idx[s:s+512]; Xb, Yb = X_tv[bi], OH[bi]
        err = softmax(Xb @ W + b) - Yb
        W -= 1.0 * (Xb.T @ err) / len(bi)
        b -= 1.0 * err.mean(0, keepdims=True)

# ── Evaluate ────────────────────────────────────
y_pred = np.argmax(softmax(X_ev @ W + b), 1)
print(f"Accuracy: {np.mean(y_pred == y_ev):.4f}\n")
print(f"{'':12}{'prec':>8}{'rec':>8}{'f1':>8}{'sup':>8}")
for i, cls in enumerate(classes):
    tp = ((y_pred==i)&(y_ev==i)).sum(); fp = ((y_pred==i)&(y_ev!=i)).sum(); fn = ((y_pred!=i)&(y_ev==i)).sum()
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
    print(f"{cls:<12}{p:>8.3f}{r:>8.3f}{2*p*r/(p+r+1e-9):>8.3f}{(y_ev==i).sum():>8}")

# ── Predict ─────────────────────────────────────
def predict(text):
    tok = tokenize(text); tf = Counter(tok); total = len(tok) or 1
    v = np.zeros((1, V))
    for w, c in tf.items():
        if w in w2i: v[0, w2i[w]] = (c / total) * idf[w]
    proba = softmax(v @ W + b)[0]
    print(f"{text!r:55} → {classes[proba.argmax()].upper()} {dict(zip(classes, proba.round(3)))}")

predict("best product. very useful")
predict("Didn't fit my 1996 Fender Strat... so its not that good")
predict("Definitely Not For The Seasoned Piano Player, what to do.")
predict("Absolutely love this guitar, sounds amazing!")
predict("It's okay, nothing special but does the job.")
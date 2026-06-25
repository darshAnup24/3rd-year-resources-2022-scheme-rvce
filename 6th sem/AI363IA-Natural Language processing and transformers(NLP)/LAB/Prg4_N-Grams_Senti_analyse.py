#nlp4.py
import numpy as np, math, re
from collections import Counter

# ── Config ──────────────────────────────────────
MIN_N, MAX_N, MIN_DF = 1, 2, 2
STOP = {'a','an','the','and','or','but','if','with','of','at','by','for','to',
        'from','in','on','is','am','are','was','were','be','been','have','has',
        'had','do','does','did','it','its','this','that','not','no','i','my',
        'we','you','he','she','they','what','which','who','how','can','will'}

# ── Data ────────────────────────────────────────
import pandas as pd

df = pd.read_csv('tweet_eval_complete.csv').dropna()

# Dataset columns:
# review,sentiment

X = df['review'].astype(str).values
y = df['sentiment'].values

np.random.seed(1)
idx = np.random.permutation(len(X))
split = int(len(X) * 0.8)

tr, te = idx[:split], idx[split:]

X_tr = [X[i] for i in tr]
X_te = [X[i] for i in te]

y_tr = [y[i] for i in tr]
y_te = [y[i] for i in te]

# ── Preprocessing + n-grams ─────────────────────
def ngrams(text):
    tok = [w for w in re.sub(r'[^a-z0-9 ]', '', text.lower()).split() if w not in STOP]
    return ['_'.join(tok[i:i+n]) for n in range(MIN_N, MAX_N+1) for i in range(len(tok)-n+1)]

X_tr_ng, X_te_ng = [ngrams(t) for t in X_tr], [ngrams(t) for t in X_te]

# ── Vocab + IDF ─────────────────────────────────
df_cnt = Counter(ng for doc in X_tr_ng for ng in set(doc))
vocab  = [ng for ng, c in df_cnt.items() if c >= MIN_DF]
w2i    = {ng: i for i, ng in enumerate(vocab)}; V = len(vocab); N = len(X_tr_ng)
idf    = {ng: math.log((N+1)/(df_cnt[ng]+1))+1 for ng in vocab}
print(f"Vocab: {V}")

# ── TF-IDF ──────────────────────────────────────
def vectorize(docs):
    M = np.zeros((len(docs), V))
    for d, ngs in enumerate(docs):
        tf = Counter(ngs); total = len(ngs) or 1
        for ng, c in tf.items():
            if ng in w2i: M[d, w2i[ng]] = (c/total) * idf[ng]
    return M

X_tv, X_ev = vectorize(X_tr_ng), vectorize(X_te_ng)

# ── Label encoding ──────────────────────────────
classes = sorted(set(y_tr)); c2i = {c:i for i,c in enumerate(classes)}; C = len(classes)
y_tv = np.array([c2i[c] for c in y_tr]); y_ev = np.array([c2i[c] for c in y_te])

# ── Logistic Regression ─────────────────────────
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

W = np.zeros((V, C)); b = np.zeros((1, C))
LR, EPOCHS, BS, L2 = 0.5, 100, 256, 1e-4
OH = np.eye(C)[y_tv]

for ep in range(EPOCHS):
    idx = np.random.permutation(len(X_tv))
    for s in range(0, len(X_tv), BS):
        bi = idx[s:s+BS]; Xb, Yb = X_tv[bi], OH[bi]
        err = softmax(Xb @ W + b) - Yb
        W -= LR * (Xb.T @ err / len(bi) + L2 * W)
        b -= LR * err.mean(0, keepdims=True)
    if (ep+1) % 20 == 0:
        print(f"Ep {ep+1}: acc={np.mean(np.argmax(softmax(X_tv @ W + b), 1)==y_tv):.4f}")

# ── Evaluate ────────────────────────────────────
y_pred = np.argmax(softmax(X_ev @ W + b), 1)
print(f"\nTest Accuracy: {np.mean(y_pred==y_ev):.4f}\n")
print(f"{'':12}{'prec':>8}{'rec':>8}{'f1':>8}{'sup':>8}")
for i, cls in enumerate(classes):
    tp = ((y_pred==i)&(y_ev==i)).sum(); fp = ((y_pred==i)&(y_ev!=i)).sum(); fn = ((y_pred!=i)&(y_ev==i)).sum()
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); f1 = 2*p*r/(p+r+1e-9)
    print(f"{cls:<12}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{(y_ev==i).sum():>8}")

# ── Predict ─────────────────────────────────────
def predict(text):
    ngs = ngrams(text); tf = Counter(ngs); total = len(ngs) or 1
    v = np.zeros((1, V))
    for ng, c in tf.items():
        if ng in w2i: v[0, w2i[ng]] = (c/total) * idf[ng]
    proba = softmax(v @ W + b)[0]
    label = classes[proba.argmax()]
    print(f"{text!r:55} → {label.upper()} {dict(zip(classes, proba.round(3)))}")

print()
predict("best product. very useful")
predict("Didn't fit my 1996 Fender Strat... so its not that good")
predict("Definitely Not For The Seasoned Piano Player, what to do.")
predict("Absolutely love this guitar, sounds amazing!")
predict("It's okay, nothing special but does the job.")
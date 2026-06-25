import csv, re, math, random
from collections import defaultdict

STOPWORDS = {
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','was','are','were','be','been','being','have','has','had','do',
    'did','will','would','could','should','may','might','not','no','nor',
    'so','yet','as','than','that','this','these','those','its','their',
    'there','here','what','which','who','when','where','why','how','all',
    'any','each','more','some','just','very','s','t','can','get','if',
    'from','by','about','also','am','an','re','ve','ll','d','m',
}

def preprocess(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOPWORDS and len(w) > 1]

class TFIDFVectorizer:
    def __init__(self, min_df=2): self.min_df = min_df

    def fit(self, docs):
        N, df = len(docs), defaultdict(int)
        for doc in docs:
            for t in set(doc): df[t] += 1
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, d in df.items() if d >= self.min_df))}
        self.idf = {t: math.log((N+1)/(df[t]+1))+1 for t in self.vocab}
        self.n_features = len(self.vocab)

    def transform(self, docs):
        vocab, idf, out = self.vocab, self.idf, []
        for doc in docs:
            tf = defaultdict(int)
            for w in doc: tf[w] += 1
            total, vec, norm_sq = len(doc) or 1, {}, 0.0
            for t, cnt in tf.items():
                if t in vocab:
                    val = (cnt/total) * idf[t]
                    vec[vocab[t]] = val; norm_sq += val*val
            norm = math.sqrt(norm_sq) or 1.0
            out.append({k: v/norm for k, v in vec.items()})
        return out

class NaiveBayes:
    def __init__(self, alpha=1.0): self.alpha = alpha

    def fit(self, X, y, n_features):
        self.classes = sorted(set(y))
        n = len(y)
        self.log_priors, self.log_likes, self.log_default = {}, {}, {}
        for c in self.classes:
            idx = [i for i, lbl in enumerate(y) if lbl == c]
            self.log_priors[c] = math.log(len(idx)/n)
            feat = defaultdict(float)
            for i in idx:
                for j, v in X[i].items(): feat[j] += v
            total = sum(feat.values()) + self.alpha * n_features
            self.log_likes[c]  = {j: math.log((v+self.alpha)/total) for j, v in feat.items()}
            self.log_default[c] = math.log(self.alpha/total)

    def predict(self, X):
        return [max(self.classes, key=lambda c: self.log_priors[c] + sum(
            v * self.log_likes[c].get(j, self.log_default[c]) for j, v in x.items())) for x in X]

def metrics(y_true, y_pred):
    print(f"Accuracy: {sum(a==b for a,b in zip(y_true,y_pred))/len(y_true):.4f}")
    for c in sorted(set(y_true)):
        tp = sum(1 for a,b in zip(y_true,y_pred) if a==c and b==c)
        fp = sum(1 for a,b in zip(y_true,y_pred) if a!=c and b==c)
        fn = sum(1 for a,b in zip(y_true,y_pred) if a==c and b!=c)
        p  = tp/(tp+fp) if tp+fp else 0.0
        r  = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*p*r/(p+r) if p+r else 0.0
        print(f"  [{c}]  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

def oversample(X, y):
    random.seed(42)
    buckets = {c: [(x,l) for x,l in zip(X,y) if l==c] for c in set(y)}
    target  = max(len(b) for b in buckets.values())
    for c, b in buckets.items():
        src = b[:]
        while len(b) < target: b.append(random.choice(src))
    balanced = [item for b in buckets.values() for item in b]
    random.shuffle(balanced)
    X_new, y_new = zip(*balanced)
    return list(X_new), list(y_new)

def run(path, text_col='text', label_col='label'):
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        rows = list(csv.DictReader(f))
    tokens = [preprocess(r[text_col]) for r in rows]
    labels = [r[label_col].strip().lower() for r in rows]

    random.seed(42)
    data = list(zip(tokens, labels)); random.shuffle(data)
    cut  = int(len(data)*0.8)
    Xtr, ytr = zip(*data[:cut]); Xte, yte = zip(*data[cut:])
    Xtr, ytr, Xte, yte = list(Xtr), list(ytr), list(Xte), list(yte)

    Xtr, ytr = oversample(Xtr, ytr)

    vec = TFIDFVectorizer(min_df=2); vec.fit(Xtr)
    Xtr_v, Xte_v = vec.transform(Xtr), vec.transform(Xte)

    model = NaiveBayes(alpha=1.0)
    model.fit(Xtr_v, ytr, n_features=vec.n_features)
    metrics(yte, model.predict(Xte_v))

    predict = lambda text: model.predict(vec.transform([preprocess(text)]))[0]
    for s in [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!!!",
        "Hey, can we reschedule tomorrow's meeting to 3pm?",
        "URGENT: Your bank account has been compromised. Verify now to avoid suspension.",
        "Please find attached the quarterly report for your review.",
        "Urgent. You have 11 week free membership. In one 1,000,000 price jackpot.",
        "Did you catch the bus?",
    ]:
        print(f"  [{predict(s).upper():4s}] {s[:65]}")

if __name__ == '__main__':
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else 'emails.csv',
        sys.argv[2] if len(sys.argv) > 2 else 'text',
        sys.argv[3] if len(sys.argv) > 3 else 'label')
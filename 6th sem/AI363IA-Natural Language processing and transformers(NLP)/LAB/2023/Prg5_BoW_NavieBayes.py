import csv, re, math
from collections import defaultdict, Counter

STOPWORDS = {
    'i','me','my','we','our','you','your','he','she','it','they','them','the','a','an',
    'and','or','but','in','on','at','to','for','of','with','is','was','are','were','be',
    'been','being','have','has','had','do','did','will','would','could','should','may',
    'might','not','no','nor','so','yet','both','either','as','than','that','this','these',
    'those','its','their','there','here','what','which','who','whom','when','where','why',
    'how','all','any','each','few','more','most','other','some','such','just','very','s',
    't','can','get','got','also','if','from','by','about',
}

def preprocess(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOPWORDS]

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class BoWVectorizer:
    def fit(self, docs):
        self.vocab = {t: i for i, t in enumerate(sorted({g for d in docs for g in d}, key=str))}

    def transform(self, docs):
        v = self.vocab
        return [{v[g]: c for g, c in Counter(d).items() if g in v} for d in docs]

class NaiveBayes:
    def fit(self, X, y, alpha=1.0):
        self.classes = list(set(y))
        n, m = len(y), max(max(v, default=0) for v in X) + 1
        self.log_prior, self.log_lhood = {}, {}
        for c in self.classes:
            idx = [i for i, yi in enumerate(y) if yi == c]
            self.log_prior[c] = math.log(len(idx) / n)
            col = defaultdict(int)
            for i in idx:
                for j, val in X[i].items(): col[j] += val
            total = sum(col.values()) + alpha * m
            self.log_lhood[c] = {j: math.log((col[j] + alpha) / total) for j in range(m)}

    def predict(self, X):
        return [max(self.classes, key=lambda c: self.log_prior[c] + sum(
            val * self.log_lhood[c].get(j, 0) for j, val in vec.items())) for vec in X]

def run(path, n=1):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    reviews, labels = [r['review'] for r in rows], [r['sentiment'] for r in rows]

    ngrams = [get_ngrams(preprocess(r), n) for r in reviews]
    cut = int(len(ngrams) * 0.8)
    Xtr, Xte, ytr, yte = ngrams[:cut], ngrams[cut:], labels[:cut], labels[cut:]

    vec = BoWVectorizer()
    vec.fit(Xtr)
    Xtr_v, Xte_v = vec.transform(Xtr), vec.transform(Xte)

    model = NaiveBayes()
    model.fit(Xtr_v, ytr)
    preds = model.predict(Xte_v)
    print(f"Accuracy: {sum(a==b for a,b in zip(yte, preds))/len(yte):.4f}")

    def predict_review(text):
        return model.predict(vec.transform([get_ngrams(preprocess(text), n)]))[0]

    for review in [
        "best product. very useful",
        "Didn't fit my 1996 Fender Strat... so its not that good",
        "Definitely Not For The Seasoned Piano Player what to do.",
        "The food at the restaurant was amazing and the staff were friendly.",
        "I had a terrible dining experience. Was called and the service was awful.",
        "The restaurant's ambience was nice, but the food was just OK"
    ]:
        print(f"{predict_review(review):8s} ← {review[:60]}")

if __name__ == '__main__':
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else 'balanced_reviews.csv',
        int(sys.argv[2]) if len(sys.argv) > 2 else 1)
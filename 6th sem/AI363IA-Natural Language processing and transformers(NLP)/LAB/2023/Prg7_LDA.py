import re, math, random, csv
from collections import defaultdict

STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it',
    'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now',
    'd','ll','m','o','re','ve','y','ain','could','would','also','get','one',
    'may','us','like','got','even','back','still','well','see','go','said',
    'say','make','know','way','take','come','good','new','first','last','two',
}

def preprocess(text):
    return [w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOPWORDS and len(w) > 2]

def build_vocab(docs, min_df=1, max_df_ratio=0.95):
    df = defaultdict(int)
    for doc in docs:
        for w in set(doc): df[w] += 1
    N = len(docs)
    vocab = sorted(w for w, c in df.items() if min_df <= c <= max_df_ratio * N)
    return vocab, {w: i for i, w in enumerate(vocab)}

class LDA:
    def __init__(self, n_topics=5, alpha=0.1, beta=0.01, n_iter=500, seed=42):
        self.K, self.alpha, self.beta, self.n_iter, self.seed = n_topics, alpha, beta, n_iter, seed

    def fit(self, corpus, V):
        random.seed(self.seed)
        K, alpha, beta, D = self.K, self.alpha, self.beta, len(corpus)
        ndk = [[0]*K for _ in range(D)]
        nkw = [[0]*V for _ in range(K)]
        nk  = [0]*K
        z = []
        for d, doc in enumerate(corpus):
            zd = []
            for w in doc:
                k = random.randint(0, K-1)
                zd.append(k); ndk[d][k] += 1; nkw[k][w] += 1; nk[k] += 1
            z.append(zd)

        for it in range(self.n_iter):
            for d, doc in enumerate(corpus):
                for n, w in enumerate(doc):
                    k = z[d][n]
                    ndk[d][k] -= 1; nkw[k][w] -= 1; nk[k] -= 1
                    probs = [(ndk[d][k] + alpha) * (nkw[k][w] + beta) / (nk[k] + V*beta) for k in range(K)]
                    k = self._sample(probs)
                    z[d][n] = k; ndk[d][k] += 1; nkw[k][w] += 1; nk[k] += 1
            if (it+1) % 100 == 0: print(f"  Iteration {it+1}/{self.n_iter}")

        self.ndk, self.nkw, self.nk, self.V = ndk, nkw, nk, V
        self.phi   = [[(nkw[k][w] + beta) / (nk[k] + V*beta) for w in range(V)] for k in range(K)]
        self.theta = [[(ndk[d][k] + alpha) / ((len(corpus[d]) or 1) + K*alpha) for k in range(K)] for d in range(D)]

    @staticmethod
    def _sample(probs):
        r, cum = random.random() * sum(probs), 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return len(probs) - 1

    def top_words(self, vocab, n=10):
        return [[(vocab[w], self.phi[k][w]) for w in sorted(range(self.V), key=lambda w: self.phi[k][w], reverse=True)[:n]]
                for k in range(self.K)]

    def doc_topics(self):
        return [max(range(self.K), key=lambda k: row[k]) for row in self.theta]

def umass_coherence(top_words_per_topic, corpus, w2i, top_n=10):
    co, doc_freq = defaultdict(int), defaultdict(int)
    for doc in corpus:
        seen = set(doc)
        for w in seen: doc_freq[w] += 1
        for w1 in seen:
            for w2 in seen:
                if w1 != w2: co[(w1, w2)] += 1
    scores = []
    for topic in top_words_per_topic:
        words = [w2i[w] for w, _ in topic[:top_n] if w in w2i]
        scores.append(sum(math.log((co.get((words[m], words[l]), 0) + 1) / (doc_freq.get(words[l], 0) + 1))
                          for m in range(1, len(words)) for l in range(m)))
    return scores

def run(docs_raw, n_topics=5, n_iter=300, top_n=10):
    tokens = [preprocess(d) for d in docs_raw]
    vocab, w2i = build_vocab(tokens)
    corpus = [[w2i[w] for w in doc if w in w2i] for doc in tokens]
    corpus, docs_raw = zip(*[(c, d) for c, d in zip(corpus, docs_raw) if c])
    corpus, docs_raw = list(corpus), list(docs_raw)

    model = LDA(n_topics=n_topics, n_iter=n_iter)
    model.fit(corpus, len(vocab))

    topics    = model.top_words(vocab, n=top_n)
    coherence = umass_coherence(topics, corpus, w2i, top_n=top_n)
    doc_tids  = model.doc_topics()

    for i, topic in enumerate(topics):
        print(f"Topic {i+1:2d} (coherence={coherence[i]:.2f}): {', '.join(f'{w}({s:.3f})' for w, s in topic)}")
    for i, (doc, tid) in enumerate(zip(docs_raw[:10], doc_tids[:10])):
        print(f"Doc {i+1:3d} -> Topic {tid+1}: \"{' '.join(doc.split()[:10])}...\"")

    return model, vocab, topics

if __name__ == '__main__':
    import sys
    path     = sys.argv[1]
    n_topics = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_iter   = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    text_col = sys.argv[4] if len(sys.argv) > 4 else 'text'

    if path.endswith('.csv'):
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            docs = [r[text_col] for r in csv.DictReader(f)]
    else:
        with open(path, encoding='utf-8', errors='ignore') as f:
            docs = [l.strip() for l in f if l.strip()]

    run(docs, n_topics=n_topics, n_iter=n_iter)
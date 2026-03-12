import os
folder = "texts"  
docs = []
for file in os.listdir(folder):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(line)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, hashlib, nltk
from itertools import combinations
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt'); nltk.download('stopwords')
random.seed(42); np.random.seed(42)

file_path = "docs.csv"
text_col = "text"

def load_docs(file_path, text_col = None):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df[text_col].dropna().astype(str).tolist() if text_col and text_col in df.columns else df.iloc[:, 0].dropna().astype(str).tolist()
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding = "utf-8") as f: return [line.strip() for line in f if line.strip()]
    raise ValueError("Only .csv and .txt files are supported")

docs = load_docs(file_path, text_col)
# docs = [ "information retrieval is the process of obtaining relevant documents", "information retrieval is the process of finding relevant documents", "information retrieval systems obtain relevant documents from large collections", "search engines use information retrieval algorithms to rank documents", "search engines use ranking algorithms to rank documents", "search engines rank web documents using retrieval algorithms", "query expansion improves search results by adding related terms", "query expansion techniques improve retrieval performance","query expansion methods add similar terms to the search query", "duplicate detection identifies similar documents on the web", "near duplicate detection finds similar web documents", "duplicate document detection improves search engine indexing", "machine learning models are used for image classification", "deep learning techniques are used in computer vision", "neural networks learn patterns from large datasets" ]

stop_words = set(stopwords.words('english')); stemmer = PorterStemmer()

def preprocess(text): return [stemmer.stem(w) for w in word_tokenize(text.lower()) if w.isalnum() and w not in stop_words]

def show_table(df, title):
    plt.figure(figsize = (8, 6)); plt.axis('off')
    if df.empty: plt.title(title); plt.text(0.5, 0.5, "No data", ha = 'center', va = 'center', fontsize = 12); plt.show(); return
    plt.table(cellText = df.values, rowLabels = df.index, colLabels = df.columns, loc = 'center'); plt.title(title); plt.show()

def sim_df(mat, title):
    df = pd.DataFrame(np.round(np.asarray(mat), 3), index = [f"Doc{i}" for i in range(len(docs))], columns = [f"Doc{i}" for i in range(len(docs))])
    show_table(df, title); return df

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0; f = 2 * p * r / (p + r) if p + r else 0
    return round(p, 3), round(r, 3), round(f, 3)

def k_shingles(words, k = 2): return set([" ".join(words[i : i + k]) for i in range(len(words) - k + 1)])
def h(x, a, b): return (a * x + b) % max_shingle
def avg_precision(score_vector, relevant_ids):
    ranked = np.argsort(score_vector)[::-1]; hits, s = 0, 0
    for rank, d in enumerate(ranked, 1):
        if d in relevant_ids: hits += 1; s += hits / rank
    return s / len(relevant_ids)

processed_docs = [" ".join(preprocess(doc)) for doc in docs]
shingles = [k_shingles(preprocess(doc), 1) for doc in docs]

num_hash, max_shingle = 50, 1000
hash_funcs = [(random.randint(1, max_shingle), random.randint(0, max_shingle)) for _ in range(num_hash)]
vocab = list(set(word for doc in shingles for word in doc))
shingle_index = {w : i for i, w in enumerate(vocab)}

signature = np.full((num_hash, len(docs)), np.inf)
for d, doc in enumerate(shingles):
    for word in doc:
        idx = shingle_index[word]
        for i, (a, b) in enumerate(hash_funcs): signature[i, d] = min(signature[i, d], h(idx, a, b))
signature = signature.astype(int)

minhash_sim = np.matrix([[np.mean(signature[:, i] == signature[:, j]) for j in range(len(docs))] for i in range(len(docs))])
sim_df(minhash_sim, "MinHash Similarity Table")

def get_lsh_candidates(sig, bands):
    rows = sig.shape[0] // bands; buckets, candidates = {}, set()
    for b in range(bands):
        for d in range(sig.shape[1]):
            band = tuple(sig[b * rows : (b + 1) * rows, d]); key = hashlib.md5(str(band).encode()).hexdigest()
            buckets.setdefault((b, key), []).append(d)
    for group in buckets.values():
        if len(group) > 1:
            for pair in combinations(group, 2): candidates.add(tuple(sorted(pair)))
    return candidates

candidates = get_lsh_candidates(signature, 10)
lsh_df = pd.DataFrame([(f"Doc{i}", f"Doc{j}") for i, j in sorted(candidates)], columns = ["Document 1", "Document 2"])
show_table(lsh_df if not lsh_df.empty else pd.DataFrame({"Result" : ["No LSH candidate pairs"]}), "LSH Candidate Pairs")

vectorizer = TfidfVectorizer(); tfidf = vectorizer.fit_transform(processed_docs)
query = "information retrieval"; q_vec = vectorizer.transform([" ".join(preprocess(query))]); scores = cosine_similarity(q_vec, tfidf)[0]
top_docs = scores.argsort()[::-1][:3]; relevant = tfidf[top_docs]; non_relevant = tfidf[[i for i in range(len(docs)) if i not in top_docs]]

alpha, beta, gamma = 1, 0.75, 0.15
new_query = alpha * q_vec + beta * np.asarray(relevant.mean(axis = 0)) - gamma * np.asarray(non_relevant.mean(axis = 0))
new_scores = cosine_similarity(np.asarray(new_query), tfidf)[0]

rocchio_df = pd.DataFrame({"Document" : [f"Doc{i}" for i in range(len(docs))], "Original Score" : np.round(scores, 3), "Updated Score" : np.round(new_scores, 3)})
show_table(rocchio_df, "Rocchio Score Table")

top_k = scores.argsort()[::-1][:5]; term_freq = {}
for doc in [processed_docs[i] for i in top_k]:
    for word in doc.split(): term_freq[word] = term_freq.get(word, 0) + 1

expanded_terms = sorted(term_freq, key = term_freq.get, reverse = True)[:5]
expanded_query = " ".join(preprocess(query)) + " " + " ".join(expanded_terms)
expanded_scores = cosine_similarity(vectorizer.transform([expanded_query]), tfidf)[0]

lca_df = pd.DataFrame({"Document" : [f"Doc{i}" for i in range(len(docs))], "LCA Score" : np.round(expanded_scores, 3)})
show_table(lca_df, "LCA Score Table")

jaccard = lambda a, b : len(a & b) / len(a | b) if len(a | b) > 0 else 0
jaccard_matrix = np.matrix([[jaccard(shingles[i], shingles[j]) for j in range(len(docs))] for i in range(len(docs))])
sim_df(jaccard_matrix, "Jaccard Similarity Table")

threshold = 0.10
ground_truth = {(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs)) if float(jaccard_matrix[i, j]) >= threshold}

bucket_rows = []
for b in [5, 10, 25]:
    if num_hash % b == 0:
        cand = get_lsh_candidates(signature, b); tp = len(cand & ground_truth); fp = len(cand - ground_truth); fn = len(ground_truth - cand)
        p, r, f = prf(tp, fp, fn); bucket_rows.append([b, len(cand), p, r, f])

bucket_df = pd.DataFrame(bucket_rows, columns = ["Bucket Size", "Candidate Pairs", "Precision", "Recall", "Fscore"])
show_table(bucket_df, "Precision Recall Fscore")

original_size = len(vocab) * len(docs)
comp_rows = []
for rows_used in [10, 20, 30, 40, 50]:
    sub_sig = signature[:rows_used, :]; correct, total = 0, 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            approx = np.mean(sub_sig[:, i] == sub_sig[:, j]) >= threshold; actual = float(jaccard_matrix[i, j]) >= threshold
            correct += int(approx == actual); total += 1
    comp_rows.append([rows_used, sub_sig.size, round(sub_sig.size / original_size, 3), round(correct / total, 3)])

compression_df = pd.DataFrame(comp_rows, columns = ["Signature Rows Used", "Signature Size", "Compression Ratio", "Accuracy"])
show_table(compression_df, "Signature Size Compression")

training_queries = ["information retrieval", "query expansion", "search engines", "duplicate detection"]
query_relevance = {"information retrieval" : {0, 1, 2, 3}, "query expansion" : {6, 7, 8}, "search engines" : {3, 4, 5}, "duplicate detection" : {9, 10, 11}}
settings = [(1.0, 0.75, 0.15), (1.0, 0.50, 0.25), (1.0, 1.00, 0.50)]

map_rows = []
for a, b, g in settings:
    before_list, after_list = [], []
    for tq in training_queries:
        tq_vec = vectorizer.transform([" ".join(preprocess(tq))]); base = cosine_similarity(tq_vec, tfidf)[0]
        top = base.argsort()[::-1][:3]; rel = tfidf[top]; nonrel = tfidf[[i for i in range(len(docs)) if i not in top]]
        rq = a * tq_vec + b * np.asarray(rel.mean(axis = 0)) - g * np.asarray(nonrel.mean(axis = 0)); updated = cosine_similarity(np.asarray(rq), tfidf)[0]
        before_list.append(avg_precision(base, query_relevance[tq])); after_list.append(avg_precision(updated, query_relevance[tq]))
    mb, ma = np.mean(before_list), np.mean(after_list); change = ((ma - mb) / mb) * 100 if mb else 0
    map_rows.append([a, b, g, round(mb, 3), round(ma, 3), round(change, 3)])

map_df = pd.DataFrame(map_rows, columns = ["Alpha", "Beta", "Gamma", "MAP Before", "MAP After", "Percent Change"])
show_table(map_df, "MAP Change")

plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1); plt.imshow(np.asarray(minhash_sim), cmap = 'viridis'); plt.colorbar(); plt.title("MinHash Similarity Heatmap"); plt.xlabel("Documents"); plt.ylabel("Documents")
plt.subplot(1, 2, 2); plt.imshow(np.asarray(jaccard_matrix), cmap = 'plasma'); plt.colorbar(); plt.title("Jaccard Similarity Heatmap"); plt.xlabel("Documents"); plt.ylabel("Documents")
plt.suptitle("Similarity Comparison : MinHash vs Jaccard"); plt.show()

plt.figure(); plt.bar(["Before Rocchio","After Rocchio"],[np.mean(scores),np.mean(new_scores)],color=["blue","orange"],edgecolor="black"); plt.title("MAP Change After Rocchio",fontsize=13); plt.ylabel("MAP Score"); plt.xlabel("Method"); plt.grid(axis="y",alpha=0.3); plt.show()

plt.figure(); 
plt.plot(bucket_df["Bucket Size"],bucket_df["Precision"],marker='o',linewidth=2,markersize=7,color="blue",label="Precision");
plt.plot(bucket_df["Bucket Size"],bucket_df["Recall"],marker='s',linewidth=2,markersize=7,color="green",label="Recall");
plt.plot(bucket_df["Bucket Size"],bucket_df["Fscore"],marker='^',linewidth=2,markersize=7,color="red",label="Fscore"); 
plt.title("Precision Recall Fscore vs Bucket Size",fontsize=13); plt.xlabel("Bucket Size"); plt.ylabel("Score"); plt.grid(alpha=0.3); plt.legend(); plt.show()

plt.figure()
plt.plot(compression_df["Signature Rows Used"],compression_df["Compression Ratio"],marker='o',linewidth=2,markersize=7,color="blue",label="Compression Ratio")
plt.plot(compression_df["Signature Rows Used"],compression_df["Accuracy"],marker='s',linewidth=2,markersize=7,color="orange",label="Accuracy"); 
plt.title("Signature Compression Ratio vs Accuracy",fontsize=13); plt.xlabel("Signature Rows Used"); plt.ylabel("Value"); plt.grid(alpha=0.3); plt.legend(); plt.show()

plt.figure()
plt.plot(labels,map_df["MAP Before"],marker='o',linewidth=2,markersize=7,color="blue",label="MAP Before"); plt.plot(labels,map_df["MAP After"],marker='s',linewidth=2,markersize=7,color="orange",label="MAP After")
plt.title("MAP Before vs After Rocchio Reweighting",fontsize=13); plt.xlabel("Reweighting Settings"); plt.ylabel("MAP"); plt.grid(alpha=0.3); plt.legend(); plt.show()

plt.figure(); plt.bar(labels,map_df["Percent Change"],color="green",edgecolor="black",alpha=0.9)
plt.title("Percent Change in MAP for Different Reweighting",fontsize=13); plt.xlabel("Reweighting Settings"); plt.ylabel("Percent Change"); plt.grid(axis="y",alpha=0.3); plt.show()

print("LCA Query Before :", query)
print("LCA Query After  :", expanded_query)
print("MinHash Near-Duplicate Pairs :", [(f"Doc{i}", f"Doc{j}") for i in range(len(docs)) for j in range(i + 1, len(docs)) if float(minhash_sim[i, j]) >= 0.20] or "None")
print("Jaccard Near-Duplicate Pairs :", [(f"Doc{i}", f"Doc{j}") for i in range(len(docs)) for j in range(i + 1, len(docs)) if float(jaccard_matrix[i, j]) >= threshold] or "None")
print("LSH Candidate Pairs :", [(f"Doc{i}", f"Doc{j}") for i, j in sorted(candidates)] or "None")
print("Rocchio Top Doc Before :", f"Doc{np.argmax(scores)}")
print("Rocchio Top Doc After  :", f"Doc{np.argmax(new_scores)}")
print("LCA Top Doc After Expansion :", f"Doc{np.argmax(expanded_scores)}")

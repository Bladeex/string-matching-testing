from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
from nltk.util import ngrams
import numpy as np

# Jaccard Similarity for tokenized words
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

# Levenshtein Distance on tokenized words
def avg_levenshtein(str1, str2):
    words1, words2 = str1.lower().split(), str2.lower().split()
    if not words1 or not words2:
        return 0  # Avoid division by zero
    distances = [min(levenshtein_distance(w1, w2) for w2 in words2) for w1 in words1]
    max_word_length = max(max(map(len, words1), default=1), max(map(len, words2), default=1))
    return 1 - (sum(distances) / (len(words1) * max_word_length))

# TF-IDF Cosine Similarity
def tfidf_cosine_similarity(str1, str2, corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2] + corpus)
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return max(similarity, 0)  # Ensure similarity is non-negative

# Hybrid Similarity Function
def hybrid_similarity(str1, str2, corpus=[]):
    jaccard = jaccard_similarity(str1, str2)
    levenshtein = avg_levenshtein(str1, str2)
    tfidf = tfidf_cosine_similarity(str1, str2, corpus) if corpus else 0

    # Ensure all scores are in range 0-1
    jaccard = max(jaccard, 0)
    levenshtein = max(levenshtein, 0)
    tfidf = max(tfidf, 0)

    return (0.4 * jaccard) + (0.3 * levenshtein) + (0.3 * tfidf)

# Example Usage
corpus = ["polypropylene", "high-density polyethylene", "hdpe", "low-density polyethylene", "ldpe", "polyethylene", "plastic", "polymer"]
corpus2 = {"hdpe": ["high-density polyethylene"], "ldpe": ["low-density polyethylene"], "pp": ["polypropylene"]}
mat1 = "hdpe-high-density-polyethylene"
mat2 = "high-density-polyethylene"

similarity_score = hybrid_similarity(mat1, mat2, corpus)
print(f"Similarity Score: {similarity_score:.4f}")
import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from collections import Counter
from nltk import ngrams
from nltk.corpus import reuters
import nltk
 
# Download nltk resources if not already present
nltk.download('reuters')
nltk.download('punkt')
 
def calculate_contextual_entropy(sentence, n=4):
    """Calculates the contextual entropy of n-grams in a sentence."""
    tokens = sentence.split()
    ngram_counts = Counter(ngrams(tokens, n))
    total_ngrams = sum(ngram_counts.values())
    probabilities = [count / total_ngrams for count in ngram_counts.values()]
    return entropy(probabilities)
 
def calculate_kl_divergence(sentence, reference_distribution, n=3):
    """Calculates the KL divergence between sentence's n-gram distribution and a reference distribution."""
    tokens = sentence.split()
    ngram_counts = Counter(ngrams(tokens, n))
    total_ngrams = sum(ngram_counts.values())
    sentence_distribution = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
 
    # Calculate KL divergence
    kl_div = 0.0
    for ngram, p in sentence_distribution.items():
        q = reference_distribution.get(ngram, 1e-10)  # small value to avoid log(0)
        kl_div += p * np.log(p / q)
    return kl_div
 
def calculate_reference_distribution(corpus, n=3):
    """Creates a reference n-gram distribution from a corpus of text."""
    ngram_counts = Counter()
    for sentence in corpus:
        tokens = sentence.split()
        ngram_counts.update(ngrams(tokens, n))
    total_ngrams = sum(ngram_counts.values())
    return {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
 
def calculate_wasserstein_distance(sentence, n=2):
    """Calculates the Wasserstein distance between bigram distributions within a sentence."""
    tokens = sentence.split()
    bigrams = list(ngrams(tokens, n))
    bigram_counts = Counter(bigrams)
    total_bigrams = sum(bigram_counts.values())
    bigram_probabilities = [count / total_bigrams for count in bigram_counts.values()]
 
    # Using pairs of indices as vectors for simplicity in Wasserstein distance calculation
    indices = list(range(len(bigram_probabilities)))
    return wasserstein_distance(indices, indices, bigram_probabilities, bigram_probabilities)
 
def reorganize_text(sentences):
    """Reorganizes sentences based on contextual entropy, KL divergence, and Wasserstein distance."""
    # Get reference distribution from Reuters corpus
    reuters_sentences = [reuters.raw(fileid) for fileid in reuters.fileids()]
    reference_distribution = calculate_reference_distribution(reuters_sentences)
 
    # Calculate the ranking metrics for each sentence
    ranked_sentences = []
    for sentence in sentences:
        entropy_val = calculate_contextual_entropy(sentence)
        kl_div = calculate_kl_divergence(sentence, reference_distribution)
        wasser_dist = calculate_wasserstein_distance(sentence)
        ranked_sentences.append((sentence, entropy_val, kl_div, wasser_dist))
 
    # Sort based on entropy (desc), KL divergence (asc), and Wasserstein distance (asc)
    ranked_sentences.sort(key=lambda x: (-x[1], x[2], x[3]))
 
    # Return only the reordered sentences
    return [sentence[0] for sentence in ranked_sentences]
 
# Example usage
text = [
    "This is the first example sentence.",
    "Another example sentence to test.",
    "Yet another sentence with some complexity.",
    "The final example sentence for this test."
]
 
reorganized_text = reorganize_text(text)
print("Reorganized Text:")
for sentence in reorganized_text:
    print(sentence)
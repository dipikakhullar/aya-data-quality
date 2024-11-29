import random
import re
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import langid
from laserembeddings import Laser
from bert_score import score

# Initialize LASER for embeddings
laser = Laser()

# Helper function: Tokenize text
def tokenize_text(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text)

# Metric 1: BERTScore
def compute_bertscore(text1: str, text2: str, model_name="bert-base-uncased"):
    """
    Compute BERTScore between two texts.
    """
    P, R, F1 = score([text1], [text2], model_type=model_name, verbose=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Metric 2: Token Repetition Rate
def token_repetition_rate(text: str) -> float:
    """
    Compute token repetition rate as a measure of redundancy in the text.
    """
    tokens = tokenize_text(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0

# Metric 3: Lexical Diversity
def lexical_diversity(text: str) -> float:
    """
    Calculate the lexical diversity of a given text.
    """
    tokens = tokenize_text(text)
    return len(set(tokens)) / len(tokens) if tokens else 0

# Metric 4: Perplexity with Lightweight Language Model
def compute_perplexity(text: str, model_name="gpt2"):
    """
    Compute perplexity using a pre-trained lightweight language model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# Metric 5: Language Identification (LID)
def identify_language(text: str) -> str:
    """
    Identify the language of the text using langid.
    """
    lang, _ = langid.classify(text)
    return lang

# Metric 6: LASER Embeddings Similarity (Cross-Language Embedding)
def compute_laser_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between LASER embeddings of two texts.
    """
    emb1 = laser.embed_sentences(text1, lang="en")
    emb2 = laser.embed_sentences(text2, lang="en")
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# Metric 7: COMETKIWI-22 and Parallel Corpus Filtering
def compute_comet_quality_score(text1: str, text2: str, model_name="Unbabel/comet-kites"):
    """
    Placeholder for COMETKIWI-22-based evaluation.
    """
    return {"Score": "Implement with COMET-specific libraries"}

# Metric 8: OpusCleaner Placeholder
def opus_cleaner_placeholder(text: str):
    """
    Placeholder for OpusCleaner metric. Requires implementation-specific libraries.
    """
    return {"Score": "Requires OpusCleaner"}

# Metric 9: Google Fact Check Tool APIs Placeholder
def fact_check_placeholder(text: str):
    """
    Placeholder for Google Fact Check Tool APIs.
    """
    return {"Score": "Implement Google Fact Check"}

# Metric 10: Coh-Metrix Placeholder
def compute_coh_metrix(text: str):
    """
    Placeholder for Coh-Metrix score.
    """
    return {"Score": "Implement with Coh-Metrix"}

# Metric 11: Bicleaner and Parallel Corpus Filtering
def bicleaner_placeholder(text: str):
    """
    Placeholder for Bicleaner metric.
    """
    return {"Score": "Requires Bicleaner"}

# Unified Function: Run All Metrics
def run_all_metrics(text1: str, text2: str = None):
    """
    Run all implemented metrics on a single or pair of texts.
    """
    results = {
        "BERTScore": compute_bertscore(text1, text2) if text2 else "Not applicable",
        "Token Repetition Rate": token_repetition_rate(text1),
        "Lexical Diversity": lexical_diversity(text1),
        "Perplexity": compute_perplexity(text1),
        "Language Identification": identify_language(text1),
        "LASER Similarity": compute_laser_similarity(text1, text2) if text2 else "Not applicable",
        "COMETKIWI-22": compute_comet_quality_score(text1, text2) if text2 else "Not applicable",
        "OpusCleaner": opus_cleaner_placeholder(text1),
        "Google Fact Check": fact_check_placeholder(text1),
        "Coh-Metrix": compute_coh_metrix(text1),
        "Bicleaner": bicleaner_placeholder(text1),
    }
    return results

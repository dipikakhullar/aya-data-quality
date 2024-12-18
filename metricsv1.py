from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import tqdm
from transformers import pipeline
import os

# Metric 1: Lexical Diversity
def lexical_diversity(text, tokenizer):
    """
    Compute lexical diversity as the proportion of repeated tokens
    using a multilingual model tokenizer.
    """
    tokens = tokenizer.tokenize(text)
    unique_tokens = set(tokens)
    return (len(unique_tokens) / len(tokens)) if tokens else 0


def preprocess_with_hf(tokenizer, text):
    """
    Preprocess text using Hugging Face tokenizer:
    - Tokenizes the text into subwords
    - Returns a list of tokens
    """
    tokens = tokenizer.tokenize(text)  # Automatically normalizes and tokenizes
    return tokens  # Return tokens as a list


# Metric 2: tokenized ngrams
def get_repeated_ngrams_hf(tokenizer, text, n):
    tokens = tokenizer.tokenize(text)  # Tokenize text
    ngrams_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngrams_list)
    repeated_ngrams = [
        (" ".join(ngram), count) for ngram, count in ngram_counts.items() if count > 1
    ]
    return repeated_ngrams

def get_max_ngram_and_count(ngram_list):
    if not ngram_list:
        return None, 0  # No repeated n-grams
    max_count = max(ngram_list, key=lambda x: x[1])[1]  # Get the max count
    max_ngrams = [ngram for ngram, count in ngram_list if count == max_count]  # Handle ties
    return max_ngrams, max_count


def calculate_partition_statistics(df, column_name):
    # Define percentiles to calculate
    percentiles = [0.9, 0.95, 0.99]

    # Calculate descriptive statistics
    partition_descriptions = df.groupby("aya_partition")[[column_name]].describe()

    # Calculate additional percentiles for each partition and metric
    additional_percentiles = (
        df.groupby("aya_partition")[[column_name]]
        .quantile(percentiles)
        .reset_index()
        .rename(columns={"level_1": "percentile"})
    )

    # Reshape the additional percentiles for easier interpretation
    percentile_table = additional_percentiles.pivot(
        index="aya_partition", columns="percentile", values=[column_name]
    )

    # Combine descriptive statistics and percentiles into one table
    stats_table = pd.concat(
        [partition_descriptions, percentile_table],
        axis=1,
    )
    return stats_table


def calculate_metrics_per_subset_and_save(df, column_name):
    # Define the desired order
    desired_order = ['aya_dataset', 'templated_afriqa', 'templated_afrisenti', 'templated_amharic_qa', 'templated_armenian_instruct', 'templated_bengali_news', 'templated_dutch_imdb', 'templated_hindi_headline', 'templated_hindi_news', 'templated_indic_paraphrase', 'templated_indic_sentiment', 'templated_indo_stories', 'templated_japanese_instruct', 'templated_joke_explaination', 'templated_ligurian_news', 'templated_masakhanews', 'templated_mintaka', 'templated_ntx_llm', 'templated_nusax_senti', 'templated_persian_farstail', 'templated_persian_instruct', 'templated_scirepeval', 'templated_seed_instruct', 'templated_soda', 'templated_tamil_stories', 'templated_tamil_thirukkural', 'templated_telugu_food', 'templated_telugu_jokes', 'templated_telugu_news', 'templated_telugu_poems', 'templated_telugu_riddles', 'templated_thai_pos', 'templated_thai_scb', 'templated_thai_usembassy', 'templated_thai_wikitionary', 'templated_turku_paraphrase', 'templated_ukranian_gec', 'templated_uner_llm', 'templated_urdu_news_category', 'templated_urdu_news_gen', 'templated_urdu_news_headline', 'templated_wiki_split', 'templated_xcsqa', 'templated_xlel_wd', 'templated_xwikis', 'translated_adversarial_qa', 'translated_cnn_dailymail', 'translated_dolly', 'translated_flan_coqa', 'translated_flan_cot', 'translated_flan_gem_wiki', 'translated_flan_lambada', 'translated_flan_qa', 'translated_hotpotqa', 'translated_joke_explaination', 'translated_mintaka', 'translated_mlqa', 'translated_nqopen', 'translated_paws', 'translated_piqa', 'translated_soda', 'translated_wiki_split', 'translated_wikiqa', 'translated_xlel_wd']


    # Step 1: Compute metrics for each subset
    metrics = df.groupby("subset_name").agg(
        avg_target_token_repetition_rate=(column_name, "mean"),
        p50_target_token_repetition_rate=(column_name, "median"),
        p90_target_token_repetition_rate=(column_name, lambda x: x.quantile(0.9)),
        p99_target_token_repetition_rate=(column_name, lambda x: x.quantile(0.99)),
        p25_target_token_repetition_rate=(column_name, lambda x: x.quantile(0.25)),
        p10_target_token_repetition_rate=(column_name, lambda x: x.quantile(0.10)),
        p5_target_token_repetition_rate=(column_name, lambda x: x.quantile(0.05))

    ).reset_index()

    # Step 2: Remove suffix from `subset_name` for ordering
    # Remove everything including the second-to-last underscore
    metrics["subset_base"] = metrics["subset_name"].apply(
        lambda x: "_".join(x.split("_")[:-2])
    )

    # Display the updated `subset_base` column for verification
    metrics[["subset_name", "subset_base"]]
    # Debugging: Check which entries in `subset_base` do not match `desired_order`
    missing_in_order = set(metrics["subset_base"]) - set(desired_order)
    missing_in_metrics = set(desired_order) - set(metrics["subset_base"])
    if missing_in_order:
        print(f"Warning: Subsets not in desired order: {missing_in_order}")
    if missing_in_metrics:
        print(f"Warning: Desired subsets not found in metrics: {missing_in_metrics}")

    # Step 3: Add a categorical column to enforce order
    metrics["subset_order"] = pd.Categorical(metrics["subset_base"], categories=desired_order, ordered=True)

    # Step 4: Sort the metrics dataframe based on the desired order
    metrics_sorted = metrics.sort_values("subset_order").drop(columns=["subset_base", "subset_order"])
    os.makedirs("metrics", exist_ok=True)
    output_path = os.path.join("metrics", column_name + ".csv")
    metrics_sorted.to_csv(output_path, index=False)
    # Display the sorted metrics
    return metrics_sorted, output_path


def get_examples_below_threshold(df, col, threshold, subset_col='subset_name', n=3, random_state=42):
    """
    Get up to `n` examples from each subset where column values are below the given threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column name to filter on.
        threshold (float): The threshold value.
        subset_col (str): The column representing subsets (default: 'subset_name').
        n (int): Number of examples to retrieve per subset.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary with subset names as keys and sampled DataFrames as values.
    """
    results = {}
    subsets = df[subset_col].unique()
    
    for subset in subsets:
        subset_df = df[df[subset_col] == subset]
        filtered_df = subset_df[subset_df[col] < threshold]
        
        if not filtered_df.empty:
            results[subset] = filtered_df.sample(
                min(len(filtered_df), n), random_state=random_state
            )
        else:
            results[subset] = pd.DataFrame()  # Empty if no rows meet the criteria
    
    return results


def get_examples_above_threshold(df, col, threshold, subset_col='subset_name', n=3, random_state=42):
    """
    Get up to `n` examples from each subset where column values are above the given threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column name to filter on.
        threshold (float): The threshold value.
        subset_col (str): The column representing subsets (default: 'subset_name').
        n (int): Number of examples to retrieve per subset.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary with subset names as keys and sampled DataFrames as values.
    """
    results = {}
    subsets = df[subset_col].unique()
    
    for subset in subsets:
        subset_df = df[df[subset_col] == subset]
        filtered_df = subset_df[subset_df[col] > threshold]
        
        if not filtered_df.empty:
            results[subset] = filtered_df.sample(
                min(len(filtered_df), n), random_state=random_state
            )
        else:
            results[subset] = pd.DataFrame()  # Empty if no rows meet the criteria
    
    return results



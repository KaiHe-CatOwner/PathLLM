from collections import defaultdict
import re
import math
# from rouge import Rouge 
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def create_input_format(candidates, references):
    gts = {}
    res = {}
    
    for i in range(len(candidates)):
        image_id = f'image_{i+1}'
        
        gts[image_id] = [references[i]]
        res[image_id] = [candidates[i]]
    
    return gts, res

def compute_cider_scores(candidate_list, reference_list):
    cider_scorer = Cider()

    gts, res = create_input_format(candidate_list, reference_list)
    score, _ = cider_scorer.compute_score(gts, res)
    return score

def compute_spice_scores(candidate_list, reference_list):
    spice_scorer = Spice()

    gts, res = create_input_format(candidate_list, reference_list)
    score, _ = spice_scorer.compute_score(gts, res)
    return score

def compute_bleu_scores(candidate_list, reference_list, avg=False):
    bleu_scores = []
    smoothie = SmoothingFunction().method1
    
    for candidate, references in zip(candidate_list, reference_list):
        candidate_tokens = candidate.split()
        reference_tokens = [reference.split() for reference in references]
        
        # Determine the maximum order of n-grams we can consider
        min_length = min(len(reference) for reference in reference_tokens)
        if min_length > 4:
            weights = (0.25, 0.25, 0.25, 0.25)  # Use 3-gram if reference sentences are long enough
        else:
            # Adjust weights based on the minimum length of the reference sentences
            weights = tuple(1/min_length for _ in range(min_length)) + tuple(0 for _ in range(4 - min_length))
        
        # Calculate BLEU score with dynamic weights
        bleu = sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothie)
        bleu_scores.append(bleu)
    
    if avg:
        return np.mean(bleu_scores)
    else:
        return bleu_scores


# def calculate_rouge(candidate, reference):
#     rouge = Rouge()
#     '''
#     candidate, reference: generated and ground-truth sentences
#     '''
#     scores = rouge.get_scores([candidate], reference)
#     return scores

def brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def modified_precision(candidate, references, n):
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    
    candidate_words = split_sentence(candidate, n)
    
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
            min_frequency[word] = min(max_frequency[word], candidate_words[word])
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P

def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def bleu(candidate, references, n, weights):

    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is bigger than n)"
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is smaller than n)"
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return str(bleu_result)

#BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result

#Exact match
def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
        
    if total == 0:
        return "0 (warning: length of candidate's words is 0)"
    else:
        return count / total

#F1
def calculate_prf_score(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    
    if len(candidate_words) == 0:
        return 0 ,0, 0
    elif len(reference_words) == 0:
        return 0 ,0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return precision, recall, 0
        else:
            return precision, recall, 2 * precision * recall / (precision + recall)
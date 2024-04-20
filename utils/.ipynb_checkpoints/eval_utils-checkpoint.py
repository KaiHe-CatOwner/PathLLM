from collections import defaultdict
import re
import math

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
def calculate_f1score(candidate, reference):
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
        return "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)
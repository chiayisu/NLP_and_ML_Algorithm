import numpy as np


def create_word_dict(corpus):
    vocab_dict = {}
    all_word_count = 0
    for sentence in corpus:
        for word in sentence:
            if(word not in vocab_dict):
                vocab_dict[word] = all_word_count
                all_word_count = all_word_count + 1
    return vocab_dict


def count_vocab(word_dict):
    return len(word_dict)
    

def create_co_matrix(corpus, windows_size, vocab_size, word_dict):
    co_matrix = np.zeros((vocab_size, vocab_size))
    for sentence_index in range(len(corpus)):
        for word_index in range(len(corpus[sentence_index])):
            left_most_word = word_index - windows_size
            right_most_word = word_index + windows_size
            for i in range(left_most_word, right_most_word + 1):
                if( i >=  0  and  i  !=  word_index  and  i < len (corpus[ sentence_index ])):
                    context_word = corpus[sentence_index][i]
                    center_word = corpus[sentence_index][word_index]
                    context_word_id = word_dict.get(context_word)
                    center_word_id = word_dict.get(center_word)
                    co_matrix[center_word_id][context_word_id] = co_matrix[center_word_id][context_word_id] + 1
    return co_matrix




corpus = np.array([["I", "go", "to", "school", "and", "I", "think"]])
word_dict = create_word_dict(corpus)
vocab_count = count_vocab(word_dict)
co_matrix = create_co_matrix(corpus, 1, vocab_count, word_dict)
print(word_dict)
print(co_matrix)

## Calculate Similarity between Words

import VectorSpaceModel as VSM

VectorModel = VSM.VSM()
## Similarity Score between Word I and others
query = [co_matrix[0]]
similarity_score = VectorModel.CosSimilarity(query, co_matrix)
print(similarity_score)

## show the word idnex whose similarity score is the largest
largest_score_word_index = VectorModel.FindTheLargestSimilarity()
largest_score_word_index = largest_score_word_index[0]
print(largest_score_word_index)

## show top k similar words
top_k_word_index_list = VectorModel.GetTopKMostSimilar()
print(top_k_word_index_list)

import PMI

PMI_Score = PMI.PMI(co_matrix)
pmi_matrix = PMI_Score.Calculate_PMI_Matrix()
ppmi_matrix = PMI_Score.Calculate_PPMI_Matrix()
print(pmi_matrix)
print(ppmi_matrix)







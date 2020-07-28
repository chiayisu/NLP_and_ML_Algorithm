import numpy as np

class VSM:
    def __init__(self):
        self.cosSimilarity = None
        self.inner_product = None
        
    def normalized_to_unit_vector(self, documents):
##      sum up all numbers at every documents
        power_documents = np.power(documents, 2)
        total_numbers = np.sum(power_documents, axis = 1)
        total_numbers = total_numbers[:, np.newaxis]
        sqrt_numbers = np.sqrt(total_numbers)
        normalizedVector = np.divide(documents, sqrt_numbers)
        return normalizedVector
        

    def CosSimilarity(self, query, documents):
        query = self.normalized_to_unit_vector(query)
        documents = self.normalized_to_unit_vector(documents)
        self.inner_product = np.inner(query, documents)
        return self.inner_product

    def FindTheLargestSimilarity(self):
        Largest_Similarity = np.argmax(self.inner_product, axis=1)
        return Largest_Similarity

    def GetTopKMostSimilar(self, k = 5):
        top_k_word_index_list = []
        sorted_similarity = np.argsort(self.inner_product)
        for index in range(len(sorted_similarity[0])-k, len(sorted_similarity[0])):
            top_k_word_index = sorted_similarity[0][index]
            top_k_word_index_list.append(top_k_word_index)
        return top_k_word_index_list

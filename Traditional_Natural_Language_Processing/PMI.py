import numpy as np

class PMI:
    def __init__(self, matrix):
        self.matrix = matrix
        self.all_word_count = self.calculate_all_words()
        self.word_count = self.calculate_word_count()

    def calculate_all_words(self):
        all_word_count = self.matrix.sum()
        return all_word_count
    
    def calculate_word_count(self):
        word_count = self.matrix.sum(axis = 0)
        return word_count

    def Calculate_PMI_Matrix(self):
        matrix_dim = self.matrix.shape[0]
        pmi_matrix = np.zeros((matrix_dim, matrix_dim))
        print("original",pmi_matrix )
        for row_index in range(self.matrix.shape[0]):
            for column_index in range(self.matrix.shape[1]):
                numerator = self.matrix[row_index][column_index] * self.all_word_count
                denominator = self.word_count[row_index] * self.word_count[column_index]
                if(denominator != 0):
                    pmi_matrix[row_index][column_index] = numerator / denominator
        pmi_matrix = np.log2(pmi_matrix)
        return pmi_matrix

    def Calculate_PPMI_Matrix(self):
        pmi_matrix = self.Calculate_PMI_Matrix()
        ppmi_matrix = np.maximum(0, pmi_matrix)
        return ppmi_matrix
        

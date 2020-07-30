import numpy as np
import pandas as pd
class NaiveBayes:
    def __init__(self, alpha = 1):
##      alpha : additive smoothing in naive bayes classifier
##              we initialize it with one which is Leplace smoothing 
##      prior_prob : This is the prior probability parameter in Naive Bayes,     
##                   and its shape is (# of category, 1)
##      conditional_prob : This is the conditional probabiulity term in Naive
##                         Bayes, and its shape is (# of category, D). D is 
##                         # of features.
        self.prior_prob = None
        self.conditional_prob = None
        self.num_of_cat = 0
        self.alpha = alpha
        
    def normalize_output_prob(self, prob):
        total = np.sum(prob, axis = 1)
        for i in range(prob.shape[0]):
            prob[i] /= total[i]
        return prob
        
    def train(self, features, label):
##      features : the feature for traing. The dimension is N*D.N is # of sample in training step.
##      label : the dimension in label is D * 1.
##      
        label_seires = pd.Series(label)
        class_count = label_seires.value_counts(sort=False)
        self.num_of_cat = class_count.shape[0]
        self.prior_prob= np.zeros(self.num_of_cat)
        all_count = class_count.sum()
        num_of_feature = features.shape[1]
        self.conditional_prob = np.zeros((self.num_of_cat, num_of_feature))
##      calculating prior probability
        for index in range(self.num_of_cat):
            self.prior_prob[index] = class_count[index] / all_count
        sum_of_all_tf = np.zeros(self.num_of_cat)
##      calculating conditional probability
        for each_class in range(self.num_of_cat):
            temp_tf = []
##      1. find the class in features with repect to each_class
##      2. sum up all tf with repect to each_class  
##      3. add all tf relate to each_class
##      4. calculat all words conditional probability
            for index, value in enumerate(features):
                if(label[index] == class_count.index[each_class]):
                    sum_of_all_tf[each_class] +=np.sum(value)
                    temp_tf.append(value)
            self.conditional_prob[each_class] = np.divide((np.sum(np.array(temp_tf), axis = 0) + self.alpha),
                                                       (sum_of_all_tf[each_class]+self.alpha*num_of_feature) )
        
    def predict(self, features):
## feature terms are the power for conditional probility
        result = np.zeros(features.shape[0])
        prob_result = np.zeros((features.shape[0], self.num_of_cat))
        for i in range(prob_result.shape[0]):
            for j in range(prob_result.shape[1]):
                temp = np.power(self.conditional_prob[j], features[i])
                prob_result[i][j] = np.prod(temp)
        prob_result = prob_result*self.prior_prob
        prob_result = self.normalize_output_prob(prob_result)
        result = np.argmax(prob_result, axis = 1)
        return result, prob_result
        
    

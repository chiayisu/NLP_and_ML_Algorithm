import numpy as np


class AP:
    def __init__(self, similarity, damp = 0.5, K = 10):
        self.similarity = similarity.astype(float)
        self.N = self.similarity.shape
        self.responsibility = np.zeros(self.N)
        self.availability = np.zeros(self.N)
        self.damp = damp
        self.K = K

    def Responsibility(self):
        for i in range(self.N[0]):
            for j in range(self.N[0]):
                tempS = np.copy(self.similarity[j])
                tempA = np.copy(self.availability[j])
                tempS[i] = np.NINF
                tempA[i] = np.NINF
                maxValue = np.amax(tempA) + np.amax(tempS)
                self.responsibility[i][j] =self.similarity[i][j]-maxValue
            
    def Availability(self):
        for i in range(self.N[0]):
            for j in range(self.N[0]):
                if(i != j):
                    tempR = np.copy(self.responsibility[i])
                    r_k_k = self.responsibility[i][i]
                    tempR[i] = np.NINF
                    tempR[j] = np.NINF
                    sumValue = tempR[tempR>0].sum()
                    tempA =  r_k_k - sumValue
                    if(tempA >= 0):
                        self.availability[i][j] = 0
                    else:
                        self.availability[i][j] = tempA
                else:
                    tempR = np.copy(self.responsibility[i])
                    tempR[j] = np.NINF
                    sumValue = tempR[tempR>0].sum()
                    self.availability[i][j] = sumValue
        

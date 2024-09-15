# CS305 Park University
# Assignment #5 Solution Code
# K-Means Evaluation and Extension

# The code below is copied from AIPython learnKMeans.py version 0.9.4.
# Not all of the file was included due to compatability issues and
# for the purposes of your programming assignment. Your instructions
# follow the definition of the learner class.

#
# BEGIN learnKMeans.py
#

# learnKMeans.py - k-means learning
# AIFCA Python3 code Version 0.9.4 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2022.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import Data_set, Learner, Data_from_file
import random
import matplotlib.pyplot as plt

class K_means_learner(Learner):
    def __init__(self,dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.random_initialize()

    def random_initialize(self):
        # class_counts[c] is the number of examples with class=c
        self.class_counts = [0]*self.num_classes
        # feature_sum[i][c] is the sum of the values of feature i for class c
        self.feature_sum = [[0]*self.num_classes
                            for feat in self.dataset.input_features]
        for eg in self.dataset.train:
            cl = random.randrange(self.num_classes) # assign eg to random class
            self.class_counts[cl] += 1
            for (ind,feat) in enumerate(self.dataset.input_features):
                self.feature_sum[ind][cl] += feat(eg)
        self.num_iterations = 0
        self.display(1,"Initial class counts: ",self.class_counts)

    def distance(self,cl,eg):
        """distance of the eg from the mean of the class"""
        return sum( (self.class_prediction(ind,cl)-feat(eg))**2
                         for (ind,feat) in enumerate(self.dataset.input_features))

    def class_prediction(self,feat_ind,cl):
        """prediction of the class cl on the feature with index feat_ind"""
        if self.class_counts[cl] == 0:
            return 0  # there are no examples so we can choose any value
        else:
            return self.feature_sum[feat_ind][cl]/self.class_counts[cl]
        
    def class_of_eg(self,eg):
        """class to which eg is assigned"""
        return (min((self.distance(cl,eg),cl)
                        for cl in range(self.num_classes)))[1]  
               # second element of tuple, which is a class with minimum distance

    def k_means_step(self):
        """Updates the model with one step of k-means. 
        Returns whether the assignment is stable.
        """
        new_class_counts = [0]*self.num_classes
        # feature_sum[i][c] is the sum of the values of feature i for class c
        new_feature_sum = [[0]*self.num_classes
                            for feat in self.dataset.input_features]
        for eg in self.dataset.train:
            cl = self.class_of_eg(eg)
            new_class_counts[cl] += 1
            for (ind,feat) in enumerate(self.dataset.input_features):
                new_feature_sum[ind][cl] += feat(eg)
        stable = (new_class_counts == self.class_counts) and (self.feature_sum == new_feature_sum)
        self.class_counts = new_class_counts
        self.feature_sum = new_feature_sum
        self.num_iterations += 1
        return stable
    
        
    def learn(self,n=100):
        """do n steps of k-means, or until convergence"""
        i=0
        stable = False
        while i<n and not stable:
            stable = self.k_means_step()
            i += 1
            self.display(1,"Iteration",self.num_iterations,
                             "class counts: ",self.class_counts," Stable=",stable)
        return stable

    def show_classes(self):
        """sorts the data by the class and prints in order.
        For visualizing small data sets
        """
        class_examples = [[] for i in range(self.num_classes)]
        for eg in self.dataset.train:
            class_examples[self.class_of_eg(eg)].append(eg)
        print("Class","Example",sep='\t')
        for cl in range(self.num_classes):
            for eg in class_examples[cl]:
                print(cl,*eg,sep='\t')
    
    def plot_error(self, maxstep=20):
        """Plots the sum-of-suares error as a function of the number of steps"""
        plt.ion()
        plt.xlabel("step")
        plt.ylabel("Avg sum-of-squares error")
        train_errors = []
        for i in range(maxstep):
            stable = self.learn(1)
            er = self.average_training_error()
            print('Avg Error:', er)
            train_errors.append(er)
            
        plt.plot(range(1,maxstep+1),train_errors,
                 label=str(self.num_classes)+" classes. Training set")
        plt.legend()
        plt.draw()

#
# END learnKMeans.py
#

# Below is the error function mentioned in the notes. 
    def average_training_error(self):
        tot = 0
        for eg in self.dataset.train:
            tot += self.distance(self.class_of_eg(eg), eg)
        return tot / len(self.dataset.train)
    
    
# 1.
# TODO: complete the definition of the average_silhouette_score
# function below. The silhouette score is a comparison of the
# similarity of an example to its assigned cluster vs the nearest
# non-assigned cluster and can be an alternative to the "elbow
# method" for finding the appropriate k value in k-means.
# https://en.wikipedia.org/wiki/Silhouette_(clustering)
    def average_silhouette_score(self):
        tot = 0 # sum all silhouette scores in this variable
        for eg in self.dataset.train:
            c = self.class_of_eg(eg)
            n = self.class_counts[c]
            # the value for 'a' should be an average of the distances
            # between the example 'eg' and all other members of eg's
            # assigned cluster.
            # TODO: determine this average here and assign it to 'a'
            a = 0

            # the value for 'b' should be an average of the distances
            # between the example 'eg' and all other members of the
            # closest cluster to the eg (other than the one eg is
            # already assigned to). 
            # TODO: determine this average here and assign it to 'b'
            b = 0
            
            # find the silhouette score based on a and b
            tot += 0 if a==0 and b==0 else (b-a)/max(a,b)
            
        return tot/len(self.dataset.train) # return the avg silhouette
            
                  
                

def main():
   trials = 100
   filename = '3-clust.csv'
   
   # 2. The file 3-clust.csv contains 3 very obvious clusters.
   # Find the average_training_error for k=2, 3, 4, and 5. Make
   # run many trials (use the 'trials' variable above) and average
   # your results since they can be very dependant on the starting
   # conditions.
   
   # make sure to set prob_test=0 (to load the entire training set)
   # and make target_index=-1 (to cut off the false target column
   # added to the datasets to make them work in AIPython)
   
   # TODO: also find the silhouette score for each k. Note that, unlike
   # average error, silhouette scores don't necessarily decrease
   # while k increases. What do both the silhouette and elbow methods
   # tell you about an appropriate k value for this synthetic dataset?
   
   # 3.
   # TODO: find an appropriate k value for the tripadvisor dataset
   # in your assignment folder. This dataset lists customer ratings
   # for a variety of different qualities of a particular trip. Again
   # use both the elbow method and the silhouette. Did your findings
   # surprise you or give you insight in to the dataset? Why or why
   # not?

if __name__ == '__main__':
  main()




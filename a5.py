# CS305 Park University
# Assignment #5 Solution Code
# K-Means Evaluation and Extension
# By Cyrille Tekam Tiako
# 14 Sep 2024

import random
import matplotlib.pyplot as plt
from learnProblem import Learner, Data_from_file  # Ensure learnProblem.py exists in the same directory

class K_means_learner(Learner):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.random_initialize()

    def random_initialize(self):
        """Randomly initialize the clusters and sum feature values."""
        self.class_counts = [0] * self.num_classes
        self.feature_sum = [[0] * self.num_classes
                            for feat in self.dataset.input_features]
        for eg in self.dataset.train:
            cl = random.randrange(self.num_classes)
            self.class_counts[cl] += 1
            for (ind, feat) in enumerate(self.dataset.input_features):
                self.feature_sum[ind][cl] += feat(eg)
        self.num_iterations = 0
        self.display(1, "Initial class counts: ", self.class_counts)

    def distance(self, cl, eg):
        """Calculate the distance of an example from the class centroid."""
        return sum((self.class_prediction(ind, cl) - feat(eg)) ** 2
                   for (ind, feat) in enumerate(self.dataset.input_features))

    def class_prediction(self, feat_ind, cl):
        """Return the predicted class centroid for a given feature."""
        if self.class_counts[cl] == 0:
            return 0
        else:
            return self.feature_sum[feat_ind][cl] / self.class_counts[cl]

    def class_of_eg(self, eg):
        """Return the class with the smallest distance to the example."""
        return min((self.distance(cl, eg), cl) for cl in range(self.num_classes))[1]

    def k_means_step(self):
        """Perform a single k-means step and check if cluster assignments are stable."""
        new_class_counts = [0] * self.num_classes
        new_feature_sum = [[0] * self.num_classes
                           for feat in self.dataset.input_features]
        for eg in self.dataset.train:
            cl = self.class_of_eg(eg)
            new_class_counts[cl] += 1
            for (ind, feat) in enumerate(self.dataset.input_features):
                new_feature_sum[ind][cl] += feat(eg)
        stable = (new_class_counts == self.class_counts) and (self.feature_sum == new_feature_sum)
        self.class_counts = new_class_counts
        self.feature_sum = new_feature_sum
        self.num_iterations += 1
        return stable

    def learn(self, n=100):
        """Run the k-means algorithm for n iterations or until convergence."""
        i = 0
        stable = False
        while i < n and not stable:
            stable = self.k_means_step()
            i += 1
            self.display(1, "Iteration", self.num_iterations,
                         "class counts: ", self.class_counts, "Stable=", stable)
        return stable

    def show_classes(self):
        """Display the data sorted by cluster for visualization."""
        class_examples = [[] for _ in range(self.num_classes)]
        for eg in self.dataset.train:
            class_examples[self.class_of_eg(eg)].append(eg)
        print("Class", "Example", sep='\t')
        for cl in range(self.num_classes):
            for eg in class_examples[cl]:
                print(cl, *eg, sep='\t')

    def plot_error(self, maxstep=20):
        """Plot the average sum-of-squares error over iterations."""
        plt.ion()
        plt.xlabel("Step")
        plt.ylabel("Avg sum-of-squares error")
        train_errors = []
        for i in range(maxstep):
            self.learn(1)
            er = self.average_training_error()
            print('Avg Error:', er)
            train_errors.append(er)
        plt.plot(range(1, maxstep + 1), train_errors,
                 label=str(self.num_classes) + " classes. Training set")
        plt.legend()
        plt.draw()

    def average_training_error(self):
        """Calculate the average training error."""
        total_error = 0
        for eg in self.dataset.train:
            total_error += self.distance(self.class_of_eg(eg), eg)
        return total_error / len(self.dataset.train)

    def average_silhouette_score(self):
        """Calculate the silhouette score for the clusters."""
        total_silhouette = 0
        for eg in self.dataset.train:
            c = self.class_of_eg(eg)
            n = self.class_counts[c]

            # Calculate a: avg distance between eg and other members of its own cluster
            a = sum(self.distance(c, other_eg) for other_eg in self.dataset.train
                    if self.class_of_eg(other_eg) == c) / (n - 1 if n > 1 else 1)

            # Calculate b: avg distance between eg and the nearest other cluster
            b = min(sum(self.distance(other_cl, other_eg)
                        for other_eg in self.dataset.train
                        if self.class_of_eg(other_eg) == other_cl) / self.class_counts[other_cl]
                    for other_cl in range(self.num_classes) if other_cl != c)

            total_silhouette += 0 if a == 0 and b == 0 else (b - a) / max(a, b)

        return total_silhouette / len(self.dataset.train)

def main():
    trials = 100
    filename = '3-clust.csv'  # Ensure this file exists

    dataset = Data_from_file(filename, target_index=-1, prob_test=0)

    # Try different k values
    k_values = [2, 3, 4, 5]
    for k in k_values:
        print(f"\nRunning for k = {k}")
        avg_error = 0
        avg_silhouette = 0
        for _ in range(trials):
            learner = K_means_learner(dataset, k)
            learner.learn()
            avg_error += learner.average_training_error()
            avg_silhouette += learner.average_silhouette_score()

        avg_error /= trials
        avg_silhouette /= trials
        print(f"Avg Training Error for k={k}: {avg_error}")
        print(f"Avg Silhouette Score for k={k}: {avg_silhouette}")

if __name__ == '__main__':
    main()


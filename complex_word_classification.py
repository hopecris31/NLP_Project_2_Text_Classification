"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Hope Crisafi

<HONOR CODE STATEMENT HERE>

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate, get_precision, get_recall


def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, labels = load_file(data_file)
    pred = [1] * len(words)
    evaluate(pred, labels)
    return pred


### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)
    best_threshold = 0
    best_fscore = 0
    for threshold in range(1, 13):
        dev_pred = [1 if len(word) >= threshold else 0 for word in dev_words]
        current_fscore = get_fscore(dev_pred, dev_labels)
        if current_fscore > best_fscore:
            best_fscore = current_fscore
            best_threshold = threshold
    best_train_pred = [1 if len(word) >= best_threshold else 0 for word in train_words]
    best_dev_pred = [1 if len(word) >= best_threshold else 0 for word in dev_words]
    print("Best threshold:", best_threshold)
    print("Best f-score:", best_fscore)
    print('___')
    evaluate(best_train_pred, train_labels)
    print('___')
    evaluate(best_dev_pred, dev_labels)
    return best_train_pred, best_dev_pred




### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts: dict):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)
    best_threshold = 0
    best_fscore = 0
    for threshold in range(1, 5000):
        dev_pred = [1 if counts[word] >= threshold else 0 for word in dev_words]
        current_fscore = get_fscore(dev_pred, dev_labels)
        if current_fscore > best_fscore:
            best_fscore = current_fscore
            best_threshold = threshold
    best_train_pred = [1 if counts[word] >= best_threshold else 0 for word in train_words]
    best_dev_pred = [1 if counts[word] >= best_threshold else 0 for word in dev_words]
    print("Best threshold:", best_threshold)
    print("Best f-score:", best_fscore)
    print('___')
    evaluate(best_train_pred, train_labels)
    print('___')
    evaluate(best_dev_pred, dev_labels)
    return best_train_pred, best_dev_pred



### 3.1: Naive Bayes

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_features = np.array([[len(word), counts[word]] for word in train_words])
    dev_features = np.array([[len(word), counts[word]] for word in dev_words])
    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    mean = train_features.mean(axis=0)
    sd = train_features.std(axis=0)
    train_features = (train_features - mean) / sd
    dev_features = (dev_features - mean) / sd

    clf = GaussianNB()
    clf.fit(train_features, train_labels)

    train_pred = clf.predict(train_features)
    dev_pred = clf.predict(dev_features)

    print("evaluate training data")
    evaluate(train_pred, train_labels)
    print("\nevaluate development data")
    evaluate(dev_pred, dev_labels)

    return train_pred, dev_pred


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_features = np.array([[len(word), counts[word]] for word in train_words])
    dev_features = np.array([[len(word), counts[word]] for word in dev_words])
    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    mean = train_features.mean(axis=0)
    sd = train_features.std(axis=0)
    train_features = (train_features - mean) / sd
    dev_features = (dev_features - mean) / sd

    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    train_pred = clf.predict(train_features)
    dev_pred = clf.predict(dev_features)

    print("evaluate training data")
    evaluate(train_pred, train_labels)
    print("\nevaluate development data")
    evaluate(dev_pred, dev_labels)

    return train_pred, dev_pred


### 3.3: Build your own classifier

def decision_tree(training_file, development_file):
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in train_words])
    dev_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in dev_words])
    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    mean = train_features.mean(axis=0)
    sd = train_features.std(axis=0)
    train_features = (train_features - mean) / sd
    dev_features = (dev_features - mean) / sd

    clf = DecisionTreeClassifier()
    clf.fit(train_features, train_labels)

    train_pred = clf.predict(train_features)
    dev_pred = clf.predict(dev_features)

    print("evaluate training data")
    evaluate(train_pred, train_labels)
    print("\nevaluate development data")
    evaluate(dev_pred, dev_labels)

    return train_pred, dev_pred

def support_vector_machine(training_file, development_file):
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in train_words])
    dev_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in dev_words])
    train_labels_np = np.array(train_labels)
    dev_labels_np = np.array(dev_labels)

    mean = train_features.mean(axis=0)
    sd = train_features.std(axis=0)
    train_features_scaled = (train_features - mean) / sd
    dev_features = (dev_features - mean) / sd

    clf = SVC()
    clf.fit(train_features_scaled, train_labels_np)

    train_pred = clf.predict(train_features_scaled)
    dev_pred = clf.predict(dev_features)

    correctly_classified = set()
    incorrectly_classified = set()
    for i, word in enumerate(train_pred):
        if train_pred[i] != train_labels_np[i]:
            incorrectly_classified.add(train_words[i])
        else:
            correctly_classified.add(train_words[i])
    #
    # index = 0
    # for prediction in x_pred:
    #     if x_pred[index] != train_y[index]:
    #         print(words[index])
    #     index += 1

    print("Miss-Classified Words:")
    print(incorrectly_classified)
    print("Correctly Classified Words:")
    print(correctly_classified)

    print("evaluate training data")
    evaluate(train_pred, train_labels)
    print("\nevaluate development data")
    evaluate(dev_pred, dev_labels_np)

    return train_pred, dev_pred
#get_fscore(train_pred, train_labels)

def get_fscore_word(y_pred: int, y_true: int, word: str, f1_scores_dict: dict):
    """Calculate the f-score of the predicted labels and store the result in a dictionary.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    word: string corresponding to the word for which f-score is calculated
    f1_scores_dict: dictionary to store the word and its f-score
    """
    try:
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        if precision == 0 and recall == 0:
            f1_scores_dict[word] = 0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)
            f1_scores_dict[word] = f1_score
    except:
        f1_scores_dict[word] = None
def random_forest(training_file, development_file):
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in train_words])
    dev_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in dev_words])
    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    mean = train_features.mean(axis=0)
    sd = train_features.std(axis=0)
    train_features = (train_features - mean) / sd
    dev_features = (dev_features - mean) / sd

    clf = RandomForestClassifier()
    clf.fit(train_features, train_labels)

    train_pred = clf.predict(train_features)
    dev_pred = clf.predict(dev_features)

    print("evaluate training data")
    evaluate(train_pred, train_labels)
    print("\nevaluate development data")
    evaluate(dev_pred, dev_labels)

    return train_pred, dev_pred


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier: SVM")
    print("-----------")
    support_vector_machine(training_file, development_file)

    print("\nMy classifier: Decision Tree")
    print("-----------")
    decision_tree(training_file, development_file)

    print("\nMy classifier: Random Forest")
    print("-----------")
    random_forest(training_file, development_file)

if __name__ == "__main__":
    training_file = "/Users/hopecrisafi/Desktop/Natural Language Processing/data/complex_words_training.txt"
    development_file = "/Users/hopecrisafi/Desktop/Natural Language Processing/data/complex_words_development.txt"
    test_file = "/Users/hopecrisafi/Desktop/Natural Language Processing/data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "/Users/hopecrisafi/Desktop/Natural Language Processing/ngram_counts.txt"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    all_words = train_words + dev_words
    all_labels = np.array(train_labels + dev_labels)
    all_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in all_words])

    mean = all_features.mean(axis=0)
    sd = all_features.std(axis=0)

    all_features = (all_features - mean) / sd

    clf = SVC()
    clf.fit(all_features, all_labels)

    test_words, _ = load_file(test_file)
    test_features = np.array([[count_syllables(word), len(wn.synsets(word))] for word in test_words])

    test_features = (test_features - mean) / sd
    test_labels = clf.predict(test_features)

    with open("test_labels.txt", "w") as file:
        for label in test_labels:
            file.write(str(label) + "\n")

    file.close()




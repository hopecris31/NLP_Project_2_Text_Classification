"""Evaluation Metrics

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for part 1 of the project.
"""


def get_accuracy(y_pred: list, y_true: list) -> float:
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        true_positives = __get_true_positives(y_pred, y_true)
        true_negatives = __get_true_negatives(y_pred, y_true)
        accuracy = (true_positives + true_negatives) / len(y_pred)
        return accuracy
    except ZeroDivisionError:
        return 0


def get_precision(y_pred: list, y_true: list):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        true_positives = __get_true_positives(y_pred, y_true)
        false_positives = __get_false_positives(y_pred, y_true)
        return true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        return None


def get_recall(y_pred: list, y_true: list):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        true_positives = __get_true_positives(y_pred, y_true)
        false_negatives = __get_false_negatives(y_pred, y_true)
        return true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        return None


def get_fscore(y_pred: list, y_true: list):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return None


def evaluate(y_pred: list, y_true: list) -> None:
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    f_score = get_fscore(y_pred, y_true)

    print("Accuracy: {:.0f}%".format(accuracy * 100))
    print("Precision: {:.0f}%".format(precision * 100))
    print("Recall: {:.0f}%".format(recall * 100))
    print("F-score: {:.0f}%".format(f_score * 100))


################################
#   PRIVATE HELPER METHODS
################################

def __get_true_positives(y_pred: list, y_true: list) -> int:
    predicted_and_true = zip(y_pred, y_true)
    true_pos_count = sum(pred == 1 and true == 1 for pred, true in predicted_and_true)
    return true_pos_count


def __get_true_negatives(y_pred: list, y_true: list) -> int:
    predicted_and_true = zip(y_pred, y_true)
    true_neg_count = sum(pred == 0 and true == 0 for pred, true in predicted_and_true)
    return true_neg_count


def __get_false_positives(y_pred: list, y_true: list) -> int:
    predicted_and_true = zip(y_pred, y_true)
    false_pos_count = sum(pred > true for pred, true in predicted_and_true)
    return false_pos_count


def __get_false_negatives(y_pred: list, y_true: list) -> int:
    predicted_and_true = zip(y_pred, y_true)
    false_neg_count = sum(pred < true for pred, true in predicted_and_true)
    return false_neg_count

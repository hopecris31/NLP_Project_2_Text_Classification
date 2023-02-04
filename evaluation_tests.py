import testing as t
import evaluation as e

predicted = [1,1,0,0,1,0,0,1,0,0]
true = [1,1,1,1,1,0,0,0,0,0]

def test_get_accuracy(y_pred, y_true):
    actual = e.get_accuracy(y_pred, y_true)
    expected = 0.7
    return t.assert_equals('testing get_accuracy', expected, actual)

def test_get_precision(y_pred, y_true):
    actual = e.get_precision(y_pred, y_true)
    expected = 0.75
    return t.assert_equals('testing get_precision', expected, actual)

def test_get_recall(y_pred, y_true):
    actual = e.get_recall(y_pred, y_true)
    expected = 0.6
    return t.assert_equals('testing get_recall', expected, actual)

def test_get_fscore(y_pred, y_true):
    actual = e.get_fscore(y_pred, y_true)
    expected = 0.6666666666666665
    return t.assert_equals('testing get_recall', expected, actual)

def test_evaluate(y_pred, y_true):
    actual = e.evaluate(y_pred, y_true)
    expected = None
    return t.assert_equals('testing get_recall', expected, actual)

if __name__ == "__main__":
    t.start_tests('Unit Tests for the evaluation module')
    test_get_accuracy(predicted, true)
    test_get_precision(predicted, true)
    test_get_recall(predicted, true)
    test_get_fscore(predicted, true)
    test_evaluate(predicted, true)
    t.finish_tests()
import testing as t
import complex_word_classification as c



test_file_MAC = "/Users/hopecrisafi/Desktop/Natural Language Processing/data"
test_file_PC = "C:/Users/hopec/OneDrive/Desktop/data/complex_words_test_unlabeled.txt"
training_file_PC = "C:/Users/hopec/OneDrive/Desktop/data/complex_words_training.txt"
development_file_PC = "C:/Users/hopec/OneDrive/Desktop/data/complex_words_development.txt"

def test_all_complex(data_file):
    expected = []
    actual = c.all_complex(data_file)
    return t.assert_equals('testing all_complex', expected, actual)

def test_word_length_threshold(training_file, development_file):
    expected = []
    actual = c.word_length_threshold(training_file, development_file)
    return t.assert_equals('testing all_complex', expected, actual)

if __name__ == "__main__":
    t.start_tests("----TESTING----")
    #test_all_complex(dataPC)
    test_word_length_threshold(training_file_PC, development_file_PC)
    t.finish_tests()


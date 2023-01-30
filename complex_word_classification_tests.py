import testing as t
import complex_word_classification as c
#h


dataMAC = "/Users/hopecrisafi/Desktop/Natural Language Processing/data"
dataPC = "C:/Users/hopec/OneDrive/Desktop/data/complex_words_test_unlabeled.txt"

def test_all_complex(data_file):
    expected = 'n'
    actual = c.all_complex(data_file)
    return t.assert_equals('testing all_complex', expected, actual)

if __name__ == "__main__":
    t.start_tests("----TESTING----")
    test_all_complex(dataPC)
    t.finish_tests()


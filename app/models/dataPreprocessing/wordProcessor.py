import nltk
import string
import numpy as np
from typing import Union
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from models.dataPreprocessing.constants.modelConstants import ModelConstants


class WordProcessor:

    """
        Remove common English stop words and punctuation from the input text.

        Parameters:
            text(str): Input text.

        Returns:
            list: List of important words after stop word removal.
    """

    @staticmethod
    def stop_word_removal(text: str) -> list:
        # Create a set of stop words including additional custom stop words
        stop_word_set = set(stopwords.words('english') + ['``', "''"])

        # Tokenize the input text into words
        tokenized_words = nltk.word_tokenize(text)

        # Filter important words by removing stop words and punctuation
        important_words = []

        for word in tokenized_words:
            if word not in stop_word_set and word not in string.punctuation:
                important_words.append(word)

        return important_words

    """
        Generate word frequency distribution for a list of words.
    
        Parameters:
            words (list): List of words.
            count (int): Number of most common words to include in the result.
    
        Returns:
            list: List of tuples containing the most common words and their frequencies.
    """

    @staticmethod
    def get_word_frequency_dist(words: list, count: int = ModelConstants.COMMON_WORD_COUNT) -> list:
        # Create a frequency distribution of the input list of words
        word_freq_dist = nltk.FreqDist(words)

        # Return the most common words along with their frequencies, limited to the specified count
        return word_freq_dist.most_common(count)

    """
            Tokenizes the input text using TF-IDF vectorization.

            Args:
                text (str): Input text to be tokenized.

            Returns:
                Union[None, csr_matrix]: Returns a sparse matrix representing the transformed text,
                or None if there's an issue with the input.
    """

    @staticmethod
    def word_tokenizer(data: np.ndarray) -> Union[None, csr_matrix]:

        tokenizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english'
        )

        tokenizer.fit(data)
        return tokenizer.transform(data)

    """
        Custom train-test split function for sparse input data.

        Args:
            x (csr_matrix): The input feature matrix in sparse format.
            y (np.ndarray): The target labels.

        Returns:
            Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]: A tuple containing the following split datasets:
                - x_train: Training set of input features.
                - x_test: Testing set of input features.
                - y_train: Training set of target labels.
                - y_test: Testing set of target labels.
    """

    @staticmethod
    def custom_train_test_split(x: csr_matrix, y: np.ndarray) -> tuple:

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            random_state = ModelConstants.RANDOM_STATE,
            test_size = ModelConstants.TEST_SIZE,
            shuffle = True,
            stratify = y
        )

        return x_train, x_test, y_train, y_test


# Example usage
word_processor = WordProcessor()
text_data = np.array([
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ])

result_matrix = word_processor.word_tokenizer(text_data)

print("Tokenized Matrix:")
print(result_matrix)

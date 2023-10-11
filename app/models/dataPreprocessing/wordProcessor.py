import nltk
import string
from nltk.corpus import stopwords
from app.static.constants.ModelConstants import ModelConstants

"""
Remove common English stop words and punctuation from the input text.

Parameters:
text (str): Input text.

Returns:
list: List of important words after stop word removal.
"""


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


def get_word_frequency_dist(words: list, count: int = ModelConstants.COMMON_WORD_COUNT) -> list:
    # Create a frequency distribution of the input list of words
    word_freq_dist = nltk.FreqDist(words)

    # Return the most common words along with their frequencies, limited to the specified count
    return word_freq_dist.most_common(count)


# Example Usage:
input_text = "Your input text here."
imp_words = stop_word_removal(input_text)
word_freq_distribution = get_word_frequency_dist(imp_words)

print(word_freq_distribution)

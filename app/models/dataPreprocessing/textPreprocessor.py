import re
from functools import reduce

"""
clean_text_functions.py

This Python module defines a TextCleaner class with methods to process and sanitize textual data.
Each method serves a specific cleaning purpose, such as removing URLs, hashtags, mentions,
punctuation, non-ASCII characters, and redundant whitespaces from the input text. The 'clean_text'
method applies these cleaning methods in sequence to sanitize and preprocess the given text.

Usage:
- Import this module into your Python script or program.
- Create an instance of the TextCleaner class.
- Utilize the provided cleaning methods individually or use the 'clean_text' method to apply
  the entire sequence of cleaning operations to a given text string.
"""


# Converts the given string into lowercase.
def convert_to_lower(text: str) -> str:
    return text.lower()


# Remove URLs from the given text.
def remove_urls(text: str) -> str:
    return re.sub('http\S+\s*', ' ', text)


# Remove hashtags from the given text.
def remove_hashtags(text: str) -> str:
    return re.sub('#\S+', '', text)


# Remove mentions (usernames starting with '@') from the given text.
def remove_mentions(text: str) -> str:
    return re.sub('@\S+', '  ', text)


# Remove various punctuation characters from the given text.
def remove_punctuations(text: str) -> str:
    return re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)


# Remove extra whitespaces from the given text.
def remove_whitespaces(text: str) -> str:
    return re.sub('\s+', ' ', text)


# Remove non-ASCII characters from the given text.
def remove_non_ascii_characters(text: str) -> str:
    return re.sub(r'[^\x00-\x7f]', r' ', text)


# Remove 'RT' and 'cc' from the given text.
def remove_rt_and_cc(text: str) -> str:
    return re.sub('RT|cc', ' ', text)


# Apply a sequence of cleaning methods to the given text.
def clean_text(text: str) -> str:
    regex_functions = [
        convert_to_lower,
        remove_urls,
        remove_hashtags,
        remove_mentions,
        remove_non_ascii_characters,
        remove_rt_and_cc,
        remove_punctuations,
        remove_whitespaces
    ]

    return reduce(lambda t, func: func(t), regex_functions, text)


# Example Usage:
input_text = "Your # input text here."
cleaned_text = clean_text(input_text)
print(cleaned_text)

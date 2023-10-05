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

import re
from functools import reduce

class TextCleaner:

    # Remove URLs from the given text.
    def remove_urls(self, text: str) -> str:
        return re.sub('http\S+\s*', ' ', text)

    # Remove hashtags from the given text.
    def remove_hashtags(self, text: str) -> str:
        return re.sub('#\S+', '', text)

    # Remove mentions (usernames starting with '@') from the given text.
    def remove_mentions(self, text: str) -> str:
        return re.sub('@\S+', '  ', text)

    # Remove various punctuation characters from the given text.
    def remove_punctuations(self, text: str) -> str:
        return re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)

    # Remove extra whitespaces from the given text.
    def remove_whitespaces(self, text: str) -> str:
        return re.sub('\s+', ' ', text)

    # Remove non-ASCII characters from the given text.
    def remove_non_ascii_characters(self, text: str) -> str:
        return re.sub(r'[^\x00-\x7f]', r' ', text)

    # Remove 'RT' and 'cc' from the given text.
    def remove_rt_and_cc(self, text: str) -> str:
        return re.sub('RT|cc', ' ', text)

    # Apply a sequence of cleaning methods to the given text.
    def cleanText(self, text: str) -> str:
        regex_methods = [
            self.remove_urls,
            self.remove_hashtags,
            self.remove_mentions,
            self.remove_non_ascii_characters,
            self.remove_rt_and_cc,
            self.remove_punctuations,
            self.remove_whitespaces
        ]

        return reduce(lambda t, method: method(t), regex_methods, text)

# Example Usage:
# text_cleaner = TextCleaner()
# cleaned_text = text_cleaner.clean_text("Your input text here.")
# print(cleaned_text)

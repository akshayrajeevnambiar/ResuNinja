import re
from functools import reduce
from app.models.constants import TextProcessingConstants

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


class TextCleaner:
    @staticmethod
    def convert_to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_urls(text: str) -> str:
        return re.sub(TextProcessingConstants.URL_REGEX, ' ', text)

    @staticmethod
    def remove_hashtags(text: str) -> str:
        return re.sub(TextProcessingConstants.HASHTAG_REGEX, '', text)

    @staticmethod
    def remove_mentions(text: str) -> str:
        return re.sub(TextProcessingConstants.MENTION_REGEX, '  ', text)

    @staticmethod
    def remove_punctuations(text: str) -> str:
        return re.sub(TextProcessingConstants.PUNCTUATION_REGEX, ' ', text)

    @staticmethod
    def remove_whitespaces(text: str) -> str:
        return re.sub(TextProcessingConstants.WHITESPACE_REGEX, ' ', text)

    @staticmethod
    def remove_non_ascii_characters(text: str) -> str:
        return re.sub(TextProcessingConstants.NON_ASCII_REGEX, ' ', text)

    @staticmethod
    def remove_rt_and_cc(text: str) -> str:
        return re.sub(TextProcessingConstants.RT_CC_REGEX, ' ', text)

    @staticmethod
    def clean_text(text: str) -> str:
        regex_functions = [
            TextCleaner.convert_to_lower,
            TextCleaner.remove_urls,
            TextCleaner.remove_hashtags,
            TextCleaner.remove_mentions,
            TextCleaner.remove_non_ascii_characters,
            TextCleaner.remove_rt_and_cc,
            TextCleaner.remove_punctuations,
            TextCleaner.remove_whitespaces
        ]

        return reduce(lambda t, func: func(t), regex_functions, text)


# Example Usage:
input_text = "Your # input text here."
cleaned_text = TextCleaner.clean_text(input_text)
print(cleaned_text)

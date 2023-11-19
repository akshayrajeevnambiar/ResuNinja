import re

"""
   This class defines constants for various regular expressions used in text processing.
   Each constant represents a specific pattern, such as URLs, hashtags, mentions, punctuation,
   whitespaces, non-ASCII characters, and patterns for extracting skills, education, and experience sections.
"""


class TextProcessingConstants:

    # Overrides the object method to restrict value modification.
    def __setattr__(self, name, value):
        raise AttributeError("Write Error: Cannot re-write constants values ")

    # Matches URLs
    URL_REGEX = 'http\S+\s*'

    # Matches hashtags
    HASHTAG_REGEX = '#\S+'

    # Matches mentions (usernames starting with '@')
    MENTION_REGEX = '@\S+'

    # Matches various punctuation characters
    PUNCTUATION_REGEX = '[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")

    # Matches one or more whitespaces
    WHITESPACE_REGEX = '\s+'

    # Matches non-ASCII characters
    NON_ASCII_REGEX = r'[^\x00-\x7f]'

    # Matches 'RT' or 'cc'
    RT_CC_REGEX = 'RT|cc'

    # Matches skills section
    SKILLS_PATTERN_REGEX = r'\bSkills\b(.*?)(?=\w+\s*-\s*Exprience|\bExperience\b|\bEducation\b|\bCompany\b|$)'

    # Matches education section
    EDUCATION_PATTERN_REGEX = r'\bEducation\b(.*?)(?=\bSkills\b|\bCompany\b|$)'

    # Matches experience section
    EXPERIENCE_PATTERN_REGEX = r'\bCompany\b(.*?)(?=\bSkills\b|\bEducation\b|$)'

import re
from app.models.constants import TextProcessingConstants
from app.models.constants import Constants

"""
    Extracts specific sections from a given text, including skills, education, and work experience.

    Parameters:
    - text (str): The input text containing information about skills, education, and work experience.

    Returns:
    A tuple containing three elements:
    - skills (str or None): Extracted skills section from the input text.
    - education (str or None): Extracted education section from the input text.
    - experience (str or None): Extracted work experience section from the input text.

    The function uses regular expressions to identify and extract each section based on predefined patterns.

    Example:
    >>> resume_text = "Skills: Python, Java. Education: Master's in Computer Science. Company: ABC Inc."
    >>> extract_sections(resume_text)
    ('Python, Java', "Master's in Computer Science", 'ABC Inc.')

    Note:
    Ensure that the regular expressions cover various cases and variations in the input text to improve accuracy.
"""


class SectionExtractor:
    @staticmethod
    def extract_sections(text: str) -> tuple:
        # Define regular expressions for skills, education, and work experience
        skills_pattern = re.compile(TextProcessingConstants.SKILLS_PATTERN_REGEX, re.DOTALL | re.IGNORECASE)
        education_pattern = re.compile(TextProcessingConstants.EDUCATION_PATTERN_REGEX, re.DOTALL | re.IGNORECASE)
        work_experience_pattern = re.compile(TextProcessingConstants.EXPERIENCE_PATTERN_REGEX, re.DOTALL | re.IGNORECASE)

        # Extract matches
        skills_match = skills_pattern.search(text)
        education_match = education_pattern.search(text)
        work_experience_match = work_experience_pattern.search(text)

        skills = skills_match.group(Constants.TEXT_GROUP).strip() if skills_match else None
        education = education_match.group(Constants.TEXT_GROUP).strip() if education_match else None
        experience = work_experience_match.group(Constants.TEXT_GROUP).strip() if work_experience_match else None

        # Return the extracted sections
        return skills, education, experience


# Example usage of the TextExtractor class
resume_text = """
Skills * Programming Languages: Python (pandas, numpy, scipy, scikit-learn, matplotlib), Sql, Java, JavaScript/JQuery.
* Machine learning: Regression, SVM, Naïve Bayes, KNN, Random Forest, Decision Trees, Boosting techniques, Cluster Analysis,
Word Embedding, Sentiment Analysis, Natural Language processing, Dimensionality reduction, Topic Modelling (LDA, NMF), PCA & Neural Nets.
* Database Visualizations: Mysql, SqlServer, Cassandra, Hbase, ElasticSearch D3.js, DC.js, Plotly, kibana, matplotlib, ggplot, Tableau.
* Others: Regular Expression, HTML, CSS, Angular 6, Logstash, Kafka, Python Flask, Git, Docker, computer vision - Open CV and understanding of Deep learning.
Education Details
Data Science Assurance Associate
Data Science Assurance Associate - Ernst & Young LLP
Skill Details
JAVASCRIPT- Experience - 24 months
jQuery- Experience - 24 months
Python- Experience - 24 months
Company Details
"""

# Create an instance of the TextExtractor class
section_extractor = SectionExtractor()

# Call the static method to extract sections
skills_demo, education_demo, experience_demo = section_extractor.extract_sections(resume_text)

# Display the extracted sections
print("Skills:", skills_demo)
print("Education:", education_demo)
print("Experience:", experience_demo)

import pickle
import nltk
import string
import re
import numpy as np
import pandas as pd
from typing import Union
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from models.dataPreprocessing.constants.modelConstants import ModelConstants


class WordProcessor:


    @staticmethod
    def setup_data(raw_df) -> pd.DataFrame:
        raw_df['Resume'] = raw_df['Resume'].apply(lambda x: x.encode('utf-8').decode('utf-8', 'ignore'))
        raw_df['cleaned_resume'] = raw_df.Resume.apply(lambda x: WordProcessor.special_character_removal(x))
        raw_df['encoded_category'] = raw_df['Category']

        lb_encd = LabelEncoder()
        for i in ['encoded_category']:
            raw_df[i] = lb_encd.fit_transform(raw_df[i])

        x = raw_df['cleaned_resume'].values
        y = raw_df['encoded_category'].values

        print(raw_df['encoded_category'])
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word', 
            ngram_range=(1,2), 
            stop_words = "english", 
            lowercase = True, 
            max_features = 500000
        )
        
        word_vectorizerobj = word_vectorizer.fit(x)
        # # Dump the file
        pickle.dump(word_vectorizer.vocabulary_, open("tfidfvocab1.pkl", "wb"))
        wordFeatures = word_vectorizer.transform(x)
        X_train,X_test,y_train,y_test = train_test_split(
            wordFeatures,
            y,
            random_state = 42, 
            test_size = 0.2,
            shuffle=True, 
            stratify=y
        )

        return X_train,X_test,y_train,y_test,lb_encd

    @staticmethod
    def setup_resume_data(raw_resume_input):
        # Testing phase
        # transformer = TfidfTransformer()
        # loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf1.pkl", "rb")))
        tf1vocab = pickle.load(open("tfidfvocab1.pkl", "rb"))
        word_Vectorizer_new = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word', 
            ngram_range=(1,2), 
            stop_words = "english", 
            lowercase = True, 
            max_features = 500000, 
            vocabulary = tf1vocab
        )
        
        raw_resume_input['inputFileContentCleaned'] = raw_resume_input['inputFileContent'].apply(lambda x: x.encode('utf-8').decode('utf-8', 'ignore'))
        raw_resume_input['inputFileContentCleaned'] = raw_resume_input.inputFileContentCleaned.apply(lambda x: WordProcessor.special_character_removal(x))
        
        # word_Vectorizer.fit(raw_resume_input['inputFileContentCleaned'])
        word_Vectorizer_new.fit(raw_resume_input['inputFileContentCleaned'])

        output = word_Vectorizer_new.transform(raw_resume_input['inputFileContentCleaned'].to_numpy())
        return output

    """
        Cleanup non-english characters from the input text.

        Parameters:
            text(str): Input text.

        Returns:
            list: List of important words after special character removal.
    """

    @staticmethod
    def special_character_removal(resumeText: str) -> str:
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace 

        return resumeText 
    
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

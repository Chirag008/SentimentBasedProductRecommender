import pandas as pd
from numpy import *
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics.pairwise import pairwise_distances
from imblearn.over_sampling import RandomOverSampler

nltk.download('stopwords')
nltk.download('wordnet')

data_file_location = 'data/sample30.csv'


class sentimentAnalyser:
    df = pd.read_csv(data_file_location)
    processed_reviews = None
    model = None
    tfidf_converter = None

    def __init__(self):
        super()

    def __clean_data(self):

        print('Sentiment Analyser Model Builder -- Cleaning Data Step Started ... ')
        # As we can see that userCity and userProvince has 94% and 99% missing values. So we can remove these columns.
        self.df = self.df.drop(columns=['reviews_userCity', 'reviews_userProvince'])

        # We don't need to maintain two columns to represent the same thing. Dropping the id column as the product
        # name we will need to recommend.
        self.df = self.df.drop(columns=['id'])

        self.df.user_sentiment.fillna('Positive', inplace=True)

        # For the missing username, we can impute any random name. It will help us in recommendation system.
        self.df.reviews_username.fillna('Chirag Bansal', inplace=True)

        # for missing values of review title, we are putting empty string
        self.df.reviews_title.fillna('', inplace=True)

        # There are some rows for which user rating is 1 / 2 and review text is negative but user sentiment is given
        # Positive.
        # Switch the Positive to Negative sentiments for these cases
        self.df.loc[
            ((self.df.reviews_rating <= 2) & (self.df.user_sentiment == 'Positive')), 'user_sentiment'] = 'Negative'

        # There are some rows for which user rating is 4 / 5 and review text is positive but user sentiment is given
        # Negative.
        # Switch the Negative to Positive sentiments for these cases
        self.df.loc[
            ((self.df.reviews_rating >= 4) & (self.df.user_sentiment == 'Negative')), 'user_sentiment'] = 'Positive'
        print('Sentiment Analyser Model Builder -- Cleaning Data Step Completed successfully ! ')

    def __preprocess_text(self):
        print('Sentiment Analyser Model Builder -- Preprocessing Text Step Started ... ')
        # Make a new column which is combination of both review title and review
        self.df['review_combined'] = self.df['reviews_title'] + ' ' + self.df['reviews_text']

        # create an object of data preprocessor.
        preprocessor = dataPreprocessor()

        ListWords = [preprocessor.reg_exp_tokenizer(m) for m in list(self.df['review_combined'])]
        # Make words in lower case
        ListWords = [preprocessor.lower_case(m) for m in ListWords]
        # Eliminate Stop_Words
        ListWords = [preprocessor.eliminate_stop_words(m) for m in ListWords]
        # Stemming
        ListWords = [preprocessor.stemming_words(m) for m in ListWords]

        processed_reviews = [' '.join(review_token_list) for review_token_list in ListWords]
        self.processed_reviews = processed_reviews
        print('Sentiment Analyser Model Builder -- Preprocessing Text Step completed successfully  ')

    def __fit_final_model(self):
        print('Sentiment Analyser Model Builder -- Fitting Model Step started .... ')
        y = self.df['user_sentiment']
        X_train, X_test, y_train, y_test = train_test_split(self.processed_reviews, y, test_size=0.2, random_state=0)
        tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        self.tfidf_converter = tfidfconverter
        # fitting the transformer on train data
        X_train = tfidfconverter.fit_transform(X_train).toarray()

        ros = RandomOverSampler(random_state=42)
        X_ROS, y_ROS = ros.fit_resample(X_train, y_train)

        model_rf_ros = RandomForestClassifier(n_estimators=200, random_state=0)
        model_rf_ros.fit(X_ROS, y_ROS)
        self.model = model_rf_ros
        print('Sentiment Analyser Model Builder -- Fitting Model Step completed successfully !')

    def __save_final_model(self, model_location, tfidf_converter_location):
        joblib.dump(self.tfidf_converter, tfidf_converter_location)
        print('Saved TFIDF Converter object successfully at location -- ', tfidf_converter_location)
        joblib.dump(self.model, model_location)
        print('Saved sentiment analysis model successfully at location -- ', model_location)

    def build_and_save_model(self, model_location='sentiment_analysis_model.pkl',
                             tfidf_converter_location='tfidf_converter.pkl'):
        self.__clean_data()
        self.__preprocess_text()
        self.__fit_final_model()
        self.__save_final_model(model_location, tfidf_converter_location)


class productRecommender:
    df = pd.read_csv(data_file_location)
    ratings = None
    model = None

    def __init__(self):
        super()

    def __clean_data(self):
        print('Product Recommendation Model Builder -- Cleaning data step started .. ')
        # For the missing username, we can impute any random name. It will help us in recommendation system.
        self.df.reviews_username.fillna('Chirag Bansal', inplace=True)

        # Dropping duplicates
        df2 = self.df.drop_duplicates(['reviews_username', 'name', 'reviews_rating'])

        # Take mean of rating for user giving different ratings to same product
        self.ratings = df2.groupby(by=['reviews_username', 'name'])['reviews_rating'].mean().to_frame()
        self.ratings.reset_index(inplace=True)
        print('Product Recommendation Model Builder -- Cleaning data step completed successfully !')

    def __fit_final_model(self):
        print('Product Recommendation Model Builder --  Fitting model step started .. ')
        # Test and Train split of the dataset.
        train, test = train_test_split(self.ratings, test_size=0.30, random_state=31)

        # Pivot the train ratings' dataset into matrix format in which columns are products and the rows are username.
        df_pivot = train.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        ).fillna(0)

        # Copy the train dataset into dummy_train
        dummy_train = train.copy()

        # The products not rated by user is marked as 1 for prediction.
        dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)

        # Convert the dummy train dataset into matrix format.
        dummy_train = dummy_train.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        ).fillna(1)

        # Create a user-product matrix.
        df_pivot = train.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        )

        # normalize the values
        mean = np.nanmean(df_pivot, axis=1)
        df_subtracted = (df_pivot.T - mean).T

        # Creating the User Similarity Matrix using pairwise_distance function.
        user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
        user_correlation[np.isnan(user_correlation)] = 0

        user_correlation[user_correlation < 0] = 0
        user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

        user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
        self.model = user_final_rating
        print('Product Recommendation Model Builder -- Fitting model step completed successfully ! ')

    def __save_final_model(self, model_location):
        joblib.dump(self.model, model_location)
        print('Product recommendation model saved at location -- ', model_location)

    def build_and_save_model(self, model_location='product_recommendation_model.pkl'):
        self.__clean_data()
        self.__fit_final_model()
        self.__save_final_model(model_location)

    def __test_the_model(self, username):
        d = self.model.loc[username].sort_values(ascending=False)[0:20]
        print(d)


class dataPreprocessor:
    """
    Class to preprocess the text data.
    """

    TEXT_TO_REMOVE = " This review was collected as part of a promotion."

    def __init__(self):
        super()

    def remove_unwanted_promotion_text(self, sentence):
        """
        To remove the unwanted promotion text from the sentence
        """
        if self.TEXT_TO_REMOVE in sentence:
            sentence = sentence.strip().replace(self.TEXT_TO_REMOVE, '')
            return str(sentence)
        return sentence

    def tokenize_words(self, sentence):
        """
        tokenize the words using the word_tokenize
        """
        return word_tokenize(sentence)

    def reg_exp_tokenizer(self, sentence):
        """
        tokenize the sentence using regular expression
        """
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(sentence)

    def lower_case(self, words):
        """
        takes a list of words and return the list of lower case words
        """
        lower_case_words = []
        for word in words:
            lower_case_words.append(word.lower())
        return lower_case_words

    def eliminate_stop_words(self, sentence):
        """
        eliminate the stop words (english language)
        """
        stop_words = set(stopwords.words("english"))
        filtered_words = []
        for word in sentence:
            if word not in stop_words:
                filtered_words.append(word)
        return filtered_words

    def stemming_words(self, words):
        """
        takes input as list of words and return stemmed words
        """
        Ps = PorterStemmer()
        Stemmed_Words = []
        for word in words:
            Stemmed_Words.append(Ps.stem(word))
        return Stemmed_Words

    def lemmatizing_words(self, words):
        """
        takes input as list of words and return lemmatized words
        """
        lm = WordNetLemmatizer()
        Lemmatized_Words = []
        for word in words:
            Lemmatized_Words.append(lm.lemmatize(word))
        return Lemmatized_Words

    def preprocess_data(self, sentence):
        """
        takes input as a sentence and return the pre-processed tokens.
        """
        sent = self.remove_unwanted_promotion_text(sentence)
        tokens = self.reg_exp_tokenizer(sent)
        lowercase_tokens = self.lower_case(tokens)
        stop_words_free_tokens = self.eliminate_stop_words(lowercase_tokens)
        stemmed_tokens = self.stemming_words(stop_words_free_tokens)
        return stemmed_tokens


if __name__ == '__main__':
    sentiment_analyser_model_builder = sentimentAnalyser()
    product_recommender_model_builder = productRecommender()

    sentiment_analyser_model_builder.build_and_save_model('models/sentiment_analysis_model.pkl',
                                                          'models/tfidf_converter.pkl')
    product_recommender_model_builder.build_and_save_model('models/product_recommendation_model.pkl')

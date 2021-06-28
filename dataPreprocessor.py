import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')

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
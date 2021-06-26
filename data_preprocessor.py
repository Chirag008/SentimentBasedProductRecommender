import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



class data_preprocessor:

    def __init__(self):
        super()
        nltk.download('stopwords')
        nltk.download('wordnet')

    def tokenize_words_Sents(self, Sent):
        return word_tokenize(Sent), sent_tokenize(Sent)

    def tokenize_words(self, Sent):
        return word_tokenize(Sent)

    def RegExpTokenizer(self, Sent):
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(Sent)

    def lower_case(self, Words):
        lower_case_words = []
        for m in Words:
            lower_case_words.append(m.lower())
        return lower_case_words

    def Eliminate_Stop_Word(self, Sent):
        stop_words = set(stopwords.words("english"))
        filtered_words = []
        for w in Sent:
            if w not in stop_words:
                filtered_words.append(w)
        return filtered_words

    def Stemming_Words(self, Words):
        Ps = PorterStemmer()
        Stemmed_Words = []
        for m in Words:
            Stemmed_Words.append(Ps.stem(m))
        return Stemmed_Words

    def Lemmatizing_Words(self, Words):
        lm = WordNetLemmatizer()
        Lemmatized_Words = []
        for m in Words:
            Lemmatized_Words.append(lm.lemmatize(m))
        return Lemmatized_Words

    def preprocess_data(self, sent):
        tokens = self.RegExpTokenizer(sent)
        lowercase_tokens = self.lower_case(tokens)
        stop_words_free_tokens = self.Eliminate_Stop_Word(lowercase_tokens)
        stemmed_tokens = self.Stemming_Words(stop_words_free_tokens)
        preprocessed_tokens = self.Lemmatizing_Words(stemmed_tokens)
        return preprocessed_tokens
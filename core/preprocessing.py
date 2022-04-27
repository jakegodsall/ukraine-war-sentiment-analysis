from collections import Counter
from nltk.stem.snowball import SnowballStemmer 
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
from sklearn.preprocessing import LabelEncoder

class NewsPreprocessor:
    def __init__(self, X_train, y_train):
        self.vocab = Counter()
        self.X_train = X_train
        self.y_train = y_train
        
        self.FREQWORDS = []
        self.COMMONENOUGHWORDS = []
        
    def get_wordcount(self):
        self.X_train.str.split().apply(self.vocab.update)

    def get_freqwords(self, num_words):
        FREQWORDS = set([w for (w, wc) in self.vocab.most_common(num_words)])
    
    def get_common_enough_words(self, filter_val):
        vocab_dict = dict(self.vocab)
        for k, v in vocab_dict.items():
            if v > filter_val:
                self.COMMONENOUGHWORDS.append(k)
    
    def to_lowercase(self, doc):
            """ 
            convert all text to lowercase and remove newline characters
            """
            return doc.lower().replace("\r", " ").replace("\n", " ")
            
    def strip_punctuation(self, doc):
        return doc.translate(str.maketrans('', '', string.punctuation))
     

    def strip_html_tags(self, doc):
        """
        remove HTML tags from the text
        """
        stripped_doc = []
        for word in doc:
            soup = BeautifulSoup(word, "html.parser")
            stripped_word = soup.get_text()
            stripped_doc.append(stripped_word)
        return stripped_doc
    
    def strip_special_chars(self, doc):
        """
        remove special characters from the text
        """
        # links
        return re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)|[–«»%()]", " ", doc)

    def remove_stopwords(self, doc):
        """
        remove stopwords from the text
        """
        STOP = stopwords.words('russian')
        words = doc.split(' ')
        return ' '.join([word for word in words if word not in STOP])

    def remove_numbers(self, doc):
        """
        remove numbers from the text
        """
        return ''.join(i for i in doc if not i.isdigit())
    
    
    def remove_freqwords(self, doc):
        """
        remove the frequent words
        """
        return " ".join([word for word in str(doc).split() if word not in self.FREQWORDS])

    def remove_rarewords(self, doc):
        """
        remove the rare words
        """
        return " ".join([word for word in str(doc).split() if word in self.COMMONENOUGHWORDS])

    def stemmer(self, doc):
        stemmer = SnowballStemmer("russian")
        return ' '.join([stemmer.stem(word) for word in doc.split(' ')])

    def label_encoder(self):
        encoder = LabelEncoder()
        return encoder.fit_transform(self.y_train)

from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime, date
import json
import matplotlib.pyplot as plt


class NewsPreprocessor:
    """
    all functions are applied at the sample level.
    for a pandas.DataFrame object do
    df.apply(lambda sample: NewsProcessor.method(sample))
    """

    def __init__(self, X_train, y_train):
        self.vocab = Counter()
        self.X_train = X_train
        self.y_train = y_train

        self.FREQWORDS = []
        self.COMMONENOUGHWORDS = []

    def get_wordcount(self):
        """
        populate the object instance of vocab (collections.Counter) with words in X_train
        """
        self.X_train.str.split().apply(self.vocab.update)

    def get_freqwords(self, num_words):
        """
        get the num_words most frequent words in the object instance of vocab
        """
        FREQWORDS = set([w for (w, wc) in self.vocab.most_common(num_words)])

    def get_common_enough_words(self, filter_val):
        """
        get all words that are present more times than threshold value filter_val
        """
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


class TrainAndSave:

    """
    class for training a model, and then exporting accuracy and loss plots, and a JSON object of shape:

        json_model = {
            Date, # date of training
            Time, # time when training finished
            Name, # name of the model
            History, model.history.history object
            Model, JSON rendering of model (for reloading of model with tf.keras.models.model_from_json)
            Vocab size, # total size of vocabulary
            Embedding size, # dimensions of word embeddings
            Max words, # max number of words in a sample
            Epochs, # number of epochs trained for
            Validation split, # proportion of training data used for validation
            Training time # time taken to train model (s)
        }

    """

    def __init__(self, X_train, y_train, vocab_size, embedding_size, max_words):
        self.X_train = X_train
        self.y_train = y_train
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_words = max_words

        self.batch_size = 32
        self.num_epochs = 10
        self.val = 0.2
        self.training_time = 0

    def _fit_model(self, model, num_epochs, val):
        """
        FOR INTERNAL USE ONLY
        fit 'model' for 'num_epochs' epochs with 'val' validation split
        """
        start = time.time()
        model.fit(self.X_train,
                  self.y_train,
                  self.batch_size,
                  epochs=num_epochs,
                  validation_split=val)

        self.training_time = time.time()-start
        self.num_epochs = num_epochs
        self.val = val

    def _create_json(self, model):
        """
        FOR INTERNAL USE ONLY
        create a JSON object containing model and training information
        """
        today = date.today()
        now = datetime.now().time()

        model_json = {
            "Date": today.strftime("%d/%m/%Y"),
            "Time": now.strftime("%H:%M:%S"),
            "Name": model.name,
            "History": model.history.history,
            "Model": model.to_json(),
            "Vocab size": self.vocab_size,
            "Embedding size": self.embedding_size,
            "Max words": self.max_words,
            "Epochs": self.num_epochs,
            "Validation split:": self.val,
            "Training time": self.training_time
        }
        return model_json

    def _plot_model(self, model):
        """
        FOR INTERNAL USE ONLY
        plot accuracy and loss metrics for the trained model
        """

        num_epochs = self.num_epochs

        fig1, acc = plt.subplots(figsize=(8, 8), constrained_layout=True)
        acc.plot(model.history.history['accuracy'])
        acc.plot(model.history.history['val_accuracy'])
        acc.set_title("Accuracy", fontsize=13)
        acc.set_ylabel('Accuracy', fontsize=13)
        acc.set_xlabel('Number of epochs', fontsize=13)
        acc.set_xticks(range(0, num_epochs), range(1, num_epochs + 1))
        acc.legend(['Train', 'Validation'], loc='upper left')

        fig2, loss = plt.subplots(figsize=(8, 8), constrained_layout=True)
        loss.plot(model.history.history['loss'])
        loss.plot(model.history.history['val_loss'])
        loss.set_title("Loss", fontsize=13)
        loss.set_ylabel('Loss', fontsize=13)
        loss.set_xlabel('Number of epochs', fontsize=13)
        loss.set_xticks(range(0, num_epochs), range(1, num_epochs + 1))
        loss.legend(['Train', 'Validation'], loc='upper left')

        return fig1, fig2

    def save(self, model, num_epochs, val, plots_dir, models_dir, batch_size=32):
        """
        train the model, generate JSON and plots, and save in respective locations
        """
        self.batch_size = batch_size
        self._fit_model(model, num_epochs, val)
        model_json = self._create_json(model)

        acc, loss = self._plot_model(model)

        acc.savefig(plots_dir / (model_json['Name'] + '_acc'))
        loss.figure.savefig(plots_dir / (model_json['Name'] + '_loss'))

        with open(models_dir / (model_json["Name"] + '.json'), 'w') as json_file:
            json.dump(model_json, json_file)

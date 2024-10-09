import re
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# Cargar el modelo de spaCy
nlp = spacy.load('es_core_news_sm')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def clean_text(self, text):
        replacements = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ',
            'Ã': 'í', 'Â': '', 'Ã¼': 'ü', 'â': '', '€': '', '™': ''
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text)
        return text

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            if word is not None:
                new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        return [word.lower() for word in words if word is not None]

    def remove_punctuation(self, words):
        new_words = []
        for word in words:
            if word is not None:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
        return new_words

    def remove_stopwords(self, words):
        stop_words = set(stopwords.words('spanish'))
        return [word for word in words if word is not None and word.lower() not in stop_words]

    def spacy_lemmatize(self, text):
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ''
        
        text = self.clean_text(text)
        words = word_tokenize(text)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        
        # Volvemos a juntar las palabras y aplicamos lematización
        text = ' '.join(words)
        return self.spacy_lemmatize(text)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Si X es una cadena de texto individual, aplicamos preprocess_text directamente
        if isinstance(X, str):
            return [self.preprocess_text(X)]
        
        # Si X es una lista o Serie de pandas, aplicamos preprocess_text a cada elemento
        return X.apply(self.preprocess_text) if hasattr(X, 'apply') else [self.preprocess_text(x) for x in X]

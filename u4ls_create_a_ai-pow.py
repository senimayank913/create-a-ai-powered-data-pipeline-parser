import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

class AIPoweredDataPipelineParser:
    def __init__(self, data_file, target_column):
        self.data_file = data_file
        self.target_column = target_column
        self.data = pd.read_csv(self.data_file)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_data(self):
        self.data[self.target_column] = self.data[self.target_column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in self.stop_words]))

    def split_data(self):
        X = self.data[self.target_column]
        y = self.data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def vectorize_data(self):
        self.vectorizer = TfidfVectorizer()
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train_vec, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test_vec)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

    def parse_pipeline(self):
        self.preprocess_data()
        self.split_data()
        self.vectorize_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    parser = AIPoweredDataPipelineParser('data.csv', 'text_column')
    parser.parse_pipeline()
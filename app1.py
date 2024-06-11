from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud
from PyPDF2 import PdfFileReader
import magic
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from sklearn.impute import SimpleImputer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer



app = Flask(__name__)



def upload_file(file):
    if file.filename == '':
        raise Exception("No file selected")
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    return file_path


# Descriptive Analysis
def descriptive_analysis(data):
    return data.describe()


# Exploratory Data Analysis
def exploratory_data_analysis(data):
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=np.object).columns.tolist()
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data[col])
        plt.title(f'Countplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# Casual Analysis
def casual_analysis(data):
        # For example, let's say we want to perform a correlation analysis between variables
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
    pass


# Prescriptive Analysis
def prescriptive_analysis(data):
    # For example, let's say we want to suggest actions based on certain conditions in the data
    if 'sales' in data.columns and 'advertising_cost' in data.columns:
        # If advertising cost is high and sales are low, suggest reducing advertising expenses
        low_sales_high_advertising = data[
            (data['sales'] < data['sales'].mean()) & (data['advertising_cost'] > data['advertising_cost'].mean())]
        if not low_sales_high_advertising.empty:
            print(
                "Recommendation: Consider reducing advertising expenses as sales are low compared to high advertising costs.")
        else:
            print("Recommendation: No specific action recommended based on current data.")
    else:
        print("Prescriptive analysis can't be performed as required columns are not present in the data.")
    pass


# Predictive Analysis
def regression_analysis(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def classification_analysis(X_train, y_train, X_test, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    return results


# Text Analysis
def word_frequency_analysis(text):
    words = re.findall(r'\b\w+\b', text)
    word_freq = Counter(words)
    return word_freq


def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score


# PDF Analysis
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as f:
        pdf_reader = PdfFileReader(f)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text



def handle_missing_values(data, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed_data



def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()



class FileFormatError(Exception):
    pass



def analyze_file(file):
    try:
        
        file_path = upload_file(file)

        
        file_type = magic.Magic(mime=True)
        mime_type = file_type.from_file(file_path)

        
        if 'text' in mime_type:
            with open(file_path, 'r') as f:
                data = f.read()
        elif 'csv' in mime_type:
            data = pd.read_csv(file_path)
        elif 'excel' in mime_type:
            data = pd.read_excel(file_path)
        elif 'pdf' in mime_type:
            data = extract_text_from_pdf(file_path)
        else:
            raise FileFormatError("Unsupported file format")

        
        if isinstance(data, pd.DataFrame):
            data = data.convert_dtypes()

        # Descriptive Analysis
        descriptive_result = descriptive_analysis(data)

        # Exploratory Data Analysis
        exploratory_data_analysis(data)

        # Casual Analysis
        casual_analysis(data)

        
        prescriptive_analysis(data)

        
        if isinstance(data, pd.DataFrame):
            X = data.drop(columns=['target'])
            y = data['target']
            
            train_size = int(0.8 * len(data))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            regression_result = regression_analysis(X_train, y_train, X_test, y_test)
            classification_result = classification_analysis(X_train, y_train, X_test, y_test)

        if isinstance(data, str):
            word_freq = word_frequency_analysis(data)
            sentiment_result = sentiment_analysis(data)


        imputed_data = handle_missing_values(data)

       
        if isinstance(data, str):
            generate_wordcloud(data)

        # Error Handling
    except FileFormatError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        analyze_file(file)
        return render_template('result.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

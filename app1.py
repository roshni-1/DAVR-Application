# Importing required libraries
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import magic
from wordcloud import WordCloud
from werkzeug.utils import secure_filename
import os

# Initializing Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Helper Functions

def detect_file_type(file):
    # Detecting the type of the uploaded file using MIME type
    mime = magic.Magic(mime=True)
    file_mime_type = mime.from_buffer(file.read(1024))
    file.seek(0)  # Reset file pointer after reading
    if 'csv' in file_mime_type:
        return 'CSV'
    elif 'text' in file_mime_type:
        return 'TXT'
    elif 'pdf' in file_mime_type:
        return 'PDF'
    else:
        raise ValueError('Unsupported file type')

def get_analysis_options(file_type):
    # analysis options based on file type
    if file_type == 'CSV':
        return ['Descriptive Analysis', 'Predictive Analysis', 'Visualization Dashboard']
    elif file_type == 'TXT':
        return ['Text Summarization', 'Sentiment Analysis', 'Visualization Dashboard']
    elif file_type == 'PDF':
        return ['PDF Analysis', 'Visualization Dashboard']
    else:
        return ['Visualization Dashboard']

def perform_descriptive_analysis(file, file_type):
    #Perform descriptive analysis on the uploaded file
    if file_type == 'CSV':
        df = pd.read_csv(file)
        text_columns, numeric_columns = [], []

        for column in df.columns:
            if df[column].dtype == 'object':
                text_columns.append(column)
            elif df[column].dtype in ['int64', 'float64']:
                numeric_columns.append(column)

        analysis_result = {}

        if text_columns:
            text_analysis_result = perform_text_analysis(df[text_columns])
            analysis_result['text_analysis'] = text_analysis_result

        if numeric_columns:
            numeric_analysis_result = perform_numeric_analysis(df[numeric_columns])
            analysis_result['numeric_analysis'] = numeric_analysis_result

            outlier_info = {}
            for column in numeric_columns:
                q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                outlier_info[column] = outliers.tolist()
            analysis_result['outliers'] = outlier_info

            correlation_matrix = df.corr()
            analysis_result['correlation_matrix'] = correlation_matrix.to_html()

            missing_values_info = {}
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    missing_values_info[column] = missing_count
            analysis_result['missing_values'] = missing_values_info

        return analysis_result

    elif file_type == 'TXT':
        text = file.read().decode('utf-8')
        if text.isdigit():
            numeric_analysis_result = perform_numeric_analysis_from_text(text)
            return {'numeric_analysis': numeric_analysis_result, 'text_analysis': None}
        else:
            text_analysis_result = perform_text_analysis_from_text(text)
            return {'text_analysis': text_analysis_result, 'numeric_analysis': None}

    elif file_type == 'PDF':
        # Placeholder for PDF analysis
        return {'text_analysis': None, 'numeric_analysis': None}

    else:
        return {'text_analysis': None, 'numeric_analysis': None}

def perform_text_analysis(df):
    return "Performing text analysis..."

def perform_numeric_analysis(df):
    #numeric analysis
    return "Performing numeric analysis..."

def perform_text_analysis_from_text(text):
    """Perform text analysis on a given text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

    word_freq = FreqDist(filtered_tokens)
    keywords = [word for word, freq in word_freq.most_common(5)]

    blob = TextBlob(text)
    named_entities = blob.noun_phrases

    vectorizer = TfidfVectorizer(max_features=1000, lowercase=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-5 - 1:-1]]
        topics.append(top_words)

    summary = blob.sentences[:10]
    return {'keywords': keywords, 'named_entities': named_entities, 'topics': topics, 'summary': summary}

def perform_numeric_analysis_from_text(text):
    """Perform numeric analysis if the text contains numeric data."""
    return "Performing numeric analysis from text file content..."

def perform_predictive_analysis(file, target_variable, model_type):
    if file_type == 'CSV':
        # Read CSV file into a DataFrame
        df = pd.read_csv(file)

        # Extract the list of columns
        columns = df.columns.tolist()

        # Prepare features and target variables
        features = df.drop(columns=[target_variable])
        target = df[target_variable]

        # Train the selected model
        if model_type == 'Regression':
            model = train_regression_model(features, target)
        elif model_type == 'Classification':
            model = train_classification_model(features, target)
        elif model_type == 'Decision Tree':
            model = train_decision_tree_model(features, target)
        elif model_type == 'Gradient Boosting':
            model = train_gradient_boosting_model(features, target)
        else:
            model = None

        # Evaluate the model
        evaluation_result = evaluate_model(model, features, target)

        return {'columns': columns, 'model': model, 'evaluation_result': evaluation_result}


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


def train_regression_model(features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_classification_model(features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def train_decision_tree_model(features, target, regression=True):
    if regression:
        model = DecisionTreeRegressor()
    else:
        model = DecisionTreeClassifier()
    model.fit(features, target)
    return model

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def train_gradient_boosting_model(features, target, regression=True):
    if regression:
        model = GradientBoostingRegressor()
    else:
        model = GradientBoostingClassifier()
    model.fit(features, target)
    return model


def evaluate_model(model, features, target):
    y_pred = model.predict(features)
   if isinstance(model, LinearRegression):
        mse = mean_squared_error(target, y_pred)
        return {'mse': mse}
    elif isinstance(model, LogisticRegression):
        accuracy = accuracy_score(target, y_pred)
        return {'accuracy': accuracy}
    else:
        return {}


def perform_text_summarization(file):
    text = file.read().decode('utf-8')
    blob = TextBlob(text)
    summary = ' '.join(blob.sentences[:10])
    return summary

def perform_sentiment_analysis(file):
    text = file.read().decode('utf-8')
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment = 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'
    return f"Sentiment: {sentiment} (Score: {sentiment_score})"

def generate_visualization_dashboard(file, file_type):
    """Generate visualization dashboard for the uploaded file."""
    if file_type == 'CSV':
        df = pd.read_csv(file)
        visualizations = {}
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            visualizations[f'{column}_histogram'] = base64.b64encode(img.getvalue()).decode()
            plt.close()

        text_columns = df.select_dtypes(include='object').columns
        for column in text_columns:
            text = ' '.join(df[column].dropna().tolist())
            plt.figure(figsize=(8, 6))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud of {column}')
            plt.axis('off')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            visualizations[f'{column}_wordcloud'] = base64.b64encode(img.getvalue()).decode()
            plt.close()

        return render_template('visualization_dashboard.html', visualizations=visualizations)

    elif file_type == 'TXT':
        return 'Visualization dashboard for text files is not yet implemented.'

    elif file_type == 'PDF':
        return 'Visualization dashboard for PDF files is not yet implemented.'

    else:
        return 'Invalid file type.'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    filename = secure_filename(file.filename)

    try:
        file_type = detect_file_type(file)
        analysis_options = get_analysis_options(file_type)
        return render_template('analysis_options.html', file_type=file_type, analysis_options=analysis_options)
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(request.url)

@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        file_type = request.form['file_type']
        analysis_type = request.form['analysis_type']
        file = request.files['file']

        if analysis_type == 'Descriptive Analysis':
            result = perform_descriptive_analysis(file, file_type)
            visualization_dashboard = generate_visualization_dashboard(file, file_type)
            return render_template('analysis_results.html', result=result, visualization_dashboard=visualization_dashboard)
        elif analysis_type == 'Predictive Analysis':
            result = perform_predictive_analysis(file, file_type)
            visualization_dashboard = generate_visualization_dashboard(file, file_type)
            return render_template('analysis_results.html', result=result, visualization_dashboard=visualization_dashboard)
        elif analysis_type == 'Text Summarization':
            result = perform_text_summarization(file)
            return render_template('analysis_results.html', result=result)
        elif analysis_type == 'Sentiment Analysis':
            result = perform_sentiment_analysis(file)
            return render_template('analysis_results.html', result=result)
        elif analysis_type == 'Visualization Dashboard':
            visualization_dashboard = generate_visualization_dashboard(file, file_type)
            return render_template('visualization_dashboard.html', visualization_dashboard=visualization_dashboard)
        else:
            flash('Invalid analysis type')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



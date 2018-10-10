import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import re
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from database

    Args:
    database_filepath: filepath to read dataset provided on command line

    Returns:
    X: dataframe with text to be used as predictor variable
    Y: dataframe with target variables, 36 categories
    category_names: names of 36 categories for model output
    """
    engine_str = 'sqlite:///' + database_filepath
    engine = create_engine(engine_str)
    df = pd.read_sql_table('Table',engine)

    X = df[df.columns[1]]
    category_names = df.columns[4:]
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    """
    Normalize, lemmatize, and tokenize text

    Args:
    text: raw text data

    Returns:
    clean_tokens: prepared model input
    """

    # lower case and remove punctuation
    text = re.sub(r'[^\w\s]',' ',text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Machine learning model with multiclass output.
    Vectorize and then apply TF-IDF to the text, and use grid search for best params

    Returns:
    cv: initalized model not yet trained
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    params = {
        'clf__estimator__n_estimators': [5,10,25]
    }
    cv = GridSearchCV(pipeline, param_grid=params)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Report the f1 score, precision and recall on test set

    Args:
    model: trained model used to predict on test set
    X_test: text used as model input
    Y_test: True labels for test set
    category_names: names of 36 classes of output
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Export model to be used in web app

    Args:
    model: trained model 
    model_filepath: command line arg, saved model name
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# Disaster Response Pipeline Project

### Project Description
Web app for an emergency worker can input a new message and get classification results in several categories.

Machine Learning model trained on thousands of real messages provided by www.figure-eight.com

ETL (Extract Transform Load) data pipeline to clean dataset 

Machine Learning model demonstrates the use of scikit-learn pipelines and grid search

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

### File Description
1. process_data.py - ETL data cleaning pipeline. Merge, clean, remove duplicates, and save clean dataset to sqlite database
2. train_classifier.py - Machine learning model that processes text and then performs multi-output classification on the 36 categories. Preprocess the data with a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. Uses machine learning pipeline to vectorize and then apply TF-IDF to the text, as well as grid search to find the best parameters
3. run.py - Python Flask app which uses the trained model to make realtime predictions of message category
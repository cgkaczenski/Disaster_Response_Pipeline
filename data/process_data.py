import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from command line args

    Args:
    message_filepath: str. filepath to message.csv
    catgegories_filepath: str. filepath to categories.csv

    Return:
    df: pandas dataframe of merged csv files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])

    return df


def replace_2(i):
    """
    Replace int 2's with 1's. Scoring function requires binary input
    Needed for lambda function

    Args:
    i - int. Can be 0, 1, or 2

    Returns:
    replaces the 2's with 1
    """
    if i == 2:
        return 1
    return i
    

def clean_data(df):
	"""
	Splits the categories column into separate columns,
	converts values to binary, and drops duplicates

	Args:
	df: pandas dataframe

	Returns:
	df: cleaned dataframe
	"""
	categories = df['categories'].str.split(';',expand=True)
	row = categories.loc[0]

	#remove last 2 character: '-1' or '-0'
	category_colname = row.apply(lambda x : x[:-2])

	# rename the columns 
	categories.columns = category_colname

	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str[-1:] 
		categories[column] = categories[column].apply(lambda x: int(x))

	df = df.drop('categories',axis=1)
	df = pd.concat([df,categories],axis=1)
	df.drop_duplicates(inplace=True)

	df['related'] = df['related'].apply(replace_2)

	return df


def save_data(df, database_filename):
	"""
	Save the clean dataset into an sqlite database

	Args:
	df: cleaned dataframe
	database_filename: name of the file to be used in train_classifier.py
	"""
	engine_str = 'sqlite:///' + database_filename
	engine = create_engine(engine_str)
	df.to_sql('Table', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

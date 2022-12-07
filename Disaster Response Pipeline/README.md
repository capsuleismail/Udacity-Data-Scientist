# Disaster Response Pipeline Project (Udacity - Data Scientist Nanodegree Program)

### Table of Contents
1. Intro
2. File Descriptions
3. Installation
4. Instructions
5. Acknowledgements

## Intro
This is the 2nd Project of the Udacity's Data Scientist Nanodegree Program with [Figure Eight](https://appen.com).
In this project, we had disaster messages that has been used to build a disaster response model that can categorize messages received in real time during a disaster event.
Therefore all messages can go straight to the agency. This project also has a web application where can input messages received and get classification results. The web app includes distribution graphs.

## File Descriptions

### Folder: app
- **run.py** - script to launch web application.<br/>
- templates:<br/>
I) **go.html** - required to run the web application.<br/>
II) **master.html** - required to run the web application.

### Folder: data
- **disaster_messages.csv** -  messages provided by Figure Eight.
- **disaster_categories.csv** -  categories of the messages.
- **process_data.py** - ETL pipeline, loaded, cleaned, extracted feature and store data in SQLite database.
- **ETL Pipeline.ipynb** - JN for ETL pipeline
- **DisasterResponse.db** - processed data stored in SQlite database 

### Folder: models
- **train_classifier.py** - machine learning pipeline
- **classifier.pkl** - saved model 
- **ML Pipeline.ipynb** - JN used to EDA to prepare the train_classifier.py script

### Folder: pics
contains all screenshots about the web app

## Installation

all libraries from Anaconda, no issue for python 3.5 or more.

## Instructions

Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database:
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
- To run ML pipeline that trains classifier and saves:
          'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
Run the following command in the app's directory to run your web app.
          'python run.py'
          
Go to http://0.0.0.0:3001/

## Acknowledgements
I) [Udacity](https://www.udacity.com).<br/>
II) [Figure Eight](https://appen.com).

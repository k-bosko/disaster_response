# Disaster Response Project

In this project I built a model for an API that classifies disaster messages. The datasets provided by [Figure Eight](https://www.figure-eight.com) contain real messages sent during disaster events and their respective categories. The task was to train the supervised ML classifier to automate categorization of the new messages so that different disaster relief agencies would receive only relevant ones.

## Deployment

https://disaster-reponse-api.herokuapp.com

## Requirements

```$ pip install -r requirements.txt```

## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it as gzip pickle object <br>
        `python models/train_classifier.py data/DisasterResponse.db models/model.p.gz`

2. Run the following command to start the server locally:<br>
    `localhost=1 python run.py`

3. Go to http://0.0.0.0:3001/


## Results

**Step 1: ETL Pipeline**
* Loaded the messages and categories datasets (`disaster_messages.csv`, `disaster_categories.csv`)
* Merged the two datasets
*	Cleaned the data 
*	Saved it in a SQLite database `DisasterResponse.db`
            
**Step 2: ML Pipeline**
* Loaded data from the SQLite database
*	Split the dataset into training and test sets
*	Built a text processing and ML pipeline using NLTK and scikit-learn's Pipeline 
*	Trained and tuned the model using GridSearchCV
*	Evaluated results on the test set
*	Exported the final model as a gzip pickle file `model.p.gz`
            
**Step 3: Python Scripts**
* Converted the jupyter notebooks into python scripts `process_data.py` and `train_classifier.py`
* Refactored the code to make it modular
            
**Step 4: Flask App**
* Uploaded sql database file and pkl file with the final model to a Flask app template
* Created data visualizations in the app using Plotly

## Discussion
Because messages are being categorized into 36 different categories and the dataset contains only 26216 messages, some categories have either very small number of positive instances or after splitting the data into train/test sets and then into cross-validation sets might end up having no positive instances at all! Even the original dataset has a category "child_alone" without a single message in it (which put constrains on using certain ML models like those relying on gradient descent).

As a result we have very high **class imbalances** in this project, which influence the classification results to a large degree. Thus, with no/little positive cases, the model predicts the '0' class most of the time and is correct, which leads to high accuracy scores, but doesn't help with identification of relevant messages in this category. The accuracy score becomes a bad evaluation metric in the highly imbalanced tasks, with f1_score being a more appropriate one.

There are several ways to handle imbalanced dataset with resampling being one of the most popular. In this case, we artificially increase the instances of the underrepresented class (e.g. SMOTE technique) or downsize the instances of the overrepresented class. Also some classifiers like RandomForestClassifier in scikit-learn have class_imbalance among parameters. (I tried tuning it, but got inferior results.)

Since it is a multilabel classification task (meaning we assign simulteniously several labels to a message which are not mutually exclusive), I am not sure if we need to artificially increase the prediction of certain labels. We risk having more false positives in this case, which means bad allocation of resources especially during disasters. For a disaster response project, I believe it is more important for the model to be able to discern relevant messages from irrelevant ones and perform some high-level categorization well (e.g. earthquake vs flood).

As such, the model trained in this project does just that. It has relatively high f1_score on 'relevant', 'aid_related', 'weather_related' categories (0.88, 0.72, 0.63 respectively for positive instances) and can discriminate between earthquake/flood/storm messages well (with f1_score scores being 0.82, 0.63, 0.66 respectively). It also identifies messages related to water, shelter, food, clothing relatively well (0.69, 0.63, 0.78, 0.48). 

## Acknowledgements
This project is part of [Udacity Data Science Nanodegree Programm](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

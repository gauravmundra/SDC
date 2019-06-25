# Service Delivery Companion Text Classification App
This project is a text classification system for categorizing IDeaS client queries. 
For a query with client provided subject(not compulsary), the system tries to identify the Root Cause, Sub Category and Related Cases from database.
The classification model is a RNN model trained on tfidf and sub-models are XGBoost Classifier models trained on count vectorizer.

## Usage

###  1. Clone the Project and Install Requirements
```
cd root:
pip install -r requirements.txt
```

### 2. ETL process
```
#copy initial data to data/ics/uploads
cd app
python ics_data_clean.py
```
This will produce the dataframe `data_final.pkl` in pickle directory.

### 3. NLP and ML process
```
python ics_train_classifier.py
```
It may take a long time, and end with the trained model `model.pkl` and submodels in pickle directory.

### 4. Run the Flask Server
```
python run.py
```
This will start the web server. 

In a new web browser window, type in the following url and press enter:
```
http://localhost:3001/
```
You will see the department selection page of the webapp. Click the prefered department, type in a sentence in the input bar, and click the `Submit` button, the server will redirect you to the calssification result page.

![img](app\static\images\ics_home.png)




## Project Components

This project consists of three components:

### 1. ETL Pipeline
Implemented in `process_data.py`.
1. Loads the messages and categories datasets
2. Merges the two datasets for model training and testing
3. Cleans the data
4. Stores the cleaned data in a SQLite database

### 2. NLP and ML Pipeline
Implemented in `train_classifier.py`.
1. Loads data from the SQLite database
2. Splits the dataset into training and test sets
3. Builds a text processing pipline
4. Builds a machine learning pipeline incorporates feature extraction and transformation
5. Trains and tunes a model using GridSearchCV
6. Outputs results on the test set
7. Exports the final model as a pickle file

### 3. Flask Web App
1. The web app that visualizes the statistics of the dataset
2. Responds to user input and classify the message
3. Display the classification result




## Assumptions
1. ##### To receive accurate classification, input client provided 'Subject'.
2. Data is correctly categorized and sub-categorized.
3. ##### Each category has atleast 150 data rows and each subcategory has atleast 10 data rows.




## Possible future bugs and errors
1. Any category or subcategory have less than minimum data rows.
2. Column names are wrong in upload training data or spelling mistake in category or subcategory column in training data.
3. Model was not fully trained and whole project files require fallback.
4. ##### (For Developer) Implement of asynchronous function for training is must in future.
5. (For Developer) Number of iterations of KERAS.Sequential().fit() must be changed according to training data size.
6. (For Developer) Reduce classification time. 




## Manual Retrain Process
```
# Put all the training data in data/ics/uploads folder in proper format and structure
# go to 'app' directory
python ics_data_clean.py
python ics_train_classifier.py
```




## Folder Structure
```
SDC
│   .gitignore
│   README.md
│   requirements.txt
│   setup.py
│
├───app
│   │   ics_data_clean.py   # program that processes data
│   │   ics_keywords.py
│   │   ics_main.py
│   │   ics_save_stop_words.py
│   │   ics_train_classifier.py  # program that trains the classification model
│   │   run.py  # Flask file that runs app
│   │
│   ├───static
│   │   ├───css
│   │   └───images
│   │
│   └───templates
│           go.html  # classification result page of web app
│           ics_master.html  # ics home
│           ics_train.html
│           ics_train_post.html
│           roa_master.html
│           SDC.html  # main page of web app
│
├───data
│   ├───ics
│   │   │
│   │   └───uploads  # training data files in xlsx format
│   │
│   └───roa
└───pickle
    ├───ics
    │   │   data_final.pkl  # final cleaned data
    │   │   keys.pkl  #keys, phrases and common queries
    │   │   stop_words.pickle  # data cleaning stop words
    │   │   training_database_index.pkl  # dataframe to keep training database record
    │   │
    │   ├───categ_data  # data from subcategories
    │   │
    │   ├───dataframes  #individual dataframes
    │   │
    │   ├───factorize
    │   │
    │   ├───models
    │   │       model.pkl  # saved model 
    │   │
    │   └───submodels  #submodels
    │
    └───roa

- README.md
```

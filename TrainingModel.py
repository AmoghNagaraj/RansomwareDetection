import pandas as pd  # Used to extract data from CSV file formats
from sklearn.model_selection import train_test_split, KFold  # Allows to split the dataset into k-consecutive folds
# without shuffling of data by default and when each fold is used once as a validation set, the k-1 remaining folds
# form the training set
from sklearn.feature_extraction.text import TfidfVectorizer  # Converting text data into matrix i.e. feature extraction
from sklearn.ensemble import RandomForestClassifier  # Used for classification, feature selection and handles noise
# in the dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Accuracy of the model,
# classification report on precision, recall and F1-score
from sklearn.model_selection import GridSearchCV  # utility function in scikit-learn that performs a grid search over
# specified parameter values for an estimator
import nltk  # Natural Language Processing
from nltk.tokenize import word_tokenize  # used to divide a given string into a list of words or tokens
from nltk.corpus import stopwords  # Used to filter out the commonly used words in english language from natural
# language processing as they carry little or no meaningful information
from nltk.stem import WordNetLemmatizer  # used for lemmatization, which is a process of grouping inflected forms[
# adding morphemes to words so the grammatical and logical meaning changes] of
# a word together to their base or dictionary form
import string  # Mainly used for data cleaning i.e. the process of identifying and correcting errors, inaccuracies,
# or inconsistencies in a dataset
import matplotlib.pyplot as plt  # Used for plotting the required 2D graphs
import seaborn as sns  # Used to create scatter plot
import joblib  # used to load and save data in a variety of formats, including NumPy arrays, Python objects,
# and compressed files and also provides a caching mechanism that allows you to store the results of expensive
# computations and reuse them instead of recomputing them

nltk.download('punkt')  # Splits text into sentences and tokens i.e. words and phrases
nltk.download('stopwords')  # Helps in filtering the commonly used words in english which don't carry much meaning
nltk.download('wordnet')  # Lexical database which provides network of word relationships including synonyms,
# hyponyms, hypernyms, etc

data = pd.read_csv(
    'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial/hello_world'
    '/Book4.csv')  # Reading the CSV file

lemmatizer = WordNetLemmatizer()  # Convert words to their base form or dictionary form known as "lemma"
stop_words = set(stopwords.words('english'))  # Will create a set of stop words in English that is present in the data


def preprocess_text(text):  # Preprocessing of the data
    tokens = word_tokenize(text.lower())  # Tokens will be converted to lowercase
    tokens = [lemmatizer.lemmatize(token) for token in tokens if
              token not in stop_words and token not in string.punctuation]  # Tokenization and Lemmatization i.e.
    # words into individual tokens, and it is reduced to base form or dictionary form respectively
    return ' '.join(tokens)


data['log_message'] = data['log_message'].apply(preprocess_text)  # Applying the preprocess_text function on the log
# message

data = data.dropna()  # Removes any empty space or empty cells from the CSV

X = data['log_message']  # X is assigned to log_message column from CSV
y = data['is_ransomware']  # y is assigned to is_ransomware column from CSV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Is used to split the
# given dataset into training and testing datasets

vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)  # Adjust min_df and max_df as needed
X_train_tfidf = vectorizer.fit_transform(X_train)  # used to transform the text data into a numerical representation,
# known as Term Frequency-Inverse Document Frequency (TF-IDF) matrix.
X_test_tfidf = vectorizer.transform(X_test)  # The process is broken down into (i) Fit Transform: Where it is used to
# create a vocabulary from the training data and calculate the TF-IDF values for each word in the vocabulary,
# by doing this, there will be consistent representation of the text data. (ii) Transform: Used to apply the learning
# vocabulary and the TF-IDF values
joblib.dump(vectorizer,
            'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial'
            '/hello_world/vectorizer.pkl')  # Used to serialize and save a vectorizer object to a pickle file.

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # n_splits=5 means that the dataset will be divided into 5
# folds, shuffle=True means that the data will be shuffled before splitting, and random_state=42 ensures that the
# shuffling is reproducible.

# The provided param_grid is used to tune the hyperparameters of a machine learning model, specifically a Random
# Forest Classifier. The grid consists of two parameters:
# n_estimators: The number of trees in the forest. This parameter controls the complexity and accuracy of the model.
# The values in the grid are 100, 200, and 300. max_depth: The maximum depth of each tree. This parameter controls
# the complexity and risk of overfitting. The values in the grid are 10, 20, and None (which means no limit)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None]
}
rf_model = RandomForestClassifier(random_state=42)  # Creates a RandomForestClassifier with a random state of 42
grid_search = GridSearchCV(rf_model, param_grid, cv=kf, n_jobs=-1)  # Creates a GridSearchCV object with the
# RandomForestClassifier, a dictionary of hyperparameters (param_grid), and a KFold object (kf) for cross-validation.
# The n_jobs parameter is set to -1, which means that all available processors will be used
grid_search.fit(X_train_tfidf, y_train)  # Fits the GridSearchCV object to the training data
best_rf_model = grid_search.best_estimator_  # Gets the best-performing model from the GridSearchCV object

y_pred_test = best_rf_model.predict(X_test_tfidf)  # Predicts the labels for the test set using the best-performing
# model
accuracy_test = accuracy_score(y_test, y_pred_test)  # Calculates the accuracy of the best-performing model on the
# test set
print(f'Accuracy on test set: {accuracy_test:.2f}')  # Prints the accuracy of the best-performing model on the test set

print(classification_report(y_test, y_pred_test))  # Prints a classification report for the best-performing model

cm = confusion_matrix(y_test, y_pred_test)  # To calculate confusion matrix
plt.figure(figsize=(8, 6))  # Setting the figure size
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)  # Plot the heatmap
plt.xlabel('Predicted')  # X-axis labelling
plt.ylabel('Actual')  # Y-axis labelling
plt.title('Confusion Matrix')  # Graph title
plt.show()  # Display graph

report = classification_report(y_test, y_pred_test, output_dict=True)  # Uses the classification_report function from
# scikit-learn to generate a classification report. The report provides a summary of the precision, recall, F1 score,
# and support for each class. The output_dict=True parameter ensures that the report is returned as a dictionary.
metrics_df = pd.DataFrame(report).transpose()  # Converts the report dictionary into a Pandas DataFrame using the
# pd.DataFrame function. The transpose method is used to swap the axes of the DataFrame, so that the class labels
# become the column headers and the metrics (precision, recall, F1 score, support) become the row headers.

plt.figure(figsize=(10, 6))  # Creates a new figure with a specified size of 10 inches in width and 6 inches in height.
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')  # Plots a bar chart using the 'precision',
# 'recall', and 'f1-score' columns from the 'metrics_df' DataFrame. The 'kind' parameter is set to 'bar' to specify
# that a bar chart should be created.
plt.title('Precision, Recall, and F1-score for each class')  # Graph title
plt.xlabel('Class')  # X-axis labelling
plt.ylabel('Score')  # Y-axis labelling
plt.xticks([0, 1], ['Non-Ransomware', 'Ransomware'], rotation=0)  # Sets the x-axis tick labels to 'Non-Ransomware'
# and 'Ransomware' for the classes 0 and 1, respectively. The 'rotation' parameter is set to 0 to specify that the
# tick labels should be horizontal.
plt.legend(loc='lower right')  # Adds a legend to the plot, which will be displayed in the lower right corner.
plt.show()  # Display graph

# joblib.dump() is a function from the joblib library that is used to serialize and save an object to a file.
# best_rf_model is the object being saved, which is a Random Forest model.
joblib.dump(best_rf_model,
            'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial'
            '/hello_world/trained_model.pkl')  # This is the file path where the model will be saved. The file name
# is trained_model.pkl, which is a common extension for Pickle files.

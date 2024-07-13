import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
new_data = pd.read_csv(
    'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial/hello_world'
    '/Book3.csv')

# Preprocess text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    tokens = [lemmatizer.lemmatize(token) for token in tokens if
              token not in stop_words and token not in string.punctuation]  # Lemmatization, remove stopwords, and
    # punctuation
    return ' '.join(tokens)


new_data['log_message'] = new_data['log_message'].apply(preprocess_text)

# Load vectorizer and transform data
vectorizer = joblib.load(
    'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial/hello_world'
    '/vectorizer.pkl')
X_new_tfidf = vectorizer.transform(new_data['log_message'])

# Load trained model and make predictions
model = joblib.load(
    'C:/Users/amogh/RNSIT/AMD Contest PC/RyzenAI/ryzen-ai-sw-1.1/ryzen-ai-sw-1.1/RyzenAI-SW/tutorial/hello_world'
    '/trained_model.pkl')
predictions = model.predict(X_new_tfidf)

# Add predictions to DataFrame
new_data['predicted_is_ransomware'] = predictions

# Calculate accuracy and print classification report
accuracy = accuracy_score(new_data['is_ransomware'], predictions)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(new_data['is_ransomware'], predictions))

# Plot confusion matrix
cm = confusion_matrix(new_data['is_ransomware'], predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot precision, recall, and F1-score
report = classification_report(new_data['is_ransomware'], predictions, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title('Precision, Recall, and F1-score for each class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.xticks([0, 1], ['Non-Ransomware', 'Ransomware'], rotation=0)
plt.legend(loc='lower right')
plt.show()

# Show log messages where ransomware is detected
ransomware_detected = new_data[new_data['predicted_is_ransomware'] == 1]
print("Log messages where ransomware is detected:")
print(ransomware_detected[['log_message', 'predicted_is_ransomware']])


# Save detected ransomware log messages to PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Detected Ransomware Log Messages', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()


pdf = PDF()
pdf.add_page()

pdf.chapter_title('Log Messages where Ransomware is detected:')
for index, row in ransomware_detected.iterrows():
    log_message = row['log_message']
    pdf.chapter_body(log_message)

pdf_output_path = 'detected_ransomware_log_messages.pdf'
pdf.output(pdf_output_path)

print(f"Ransomware detected log messages have been saved to {pdf_output_path}")

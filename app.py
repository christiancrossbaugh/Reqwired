import nltk
nltk.download('wordnet')
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
import io
import sys

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Function for synonym replacement data augmentation
def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Function to train and evaluate a classifier
def train_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# Dummy data for user management
users = {
    'user1': {'username': 'user1', 'password': 'password1', 'role': 'user'},
    'admin': {'username': 'admin', 'password': 'admin123', 'role': 'admin'}
}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the machine learning tab
@app.route('/ml')
def machine_learning():
    results = {} 
    return render_template('ml.html', results=results)

# Route for uploading CSV
# Route for the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        # Save the file to a temporary location
        csv_path = "temp.csv"
        uploaded_file.save(csv_path)
        # Step 1: Read CSV and preprocess data
        df = pd.read_csv(csv_path, delimiter=',')
        # Assuming the CSV contains two columns: 'requirement' and 'reqLabel'
        X = df['requirement']
        y = df['reqLabel']

        # Determine the minority class
        minority_class = y.value_counts().idxmin()

        # Data augmentation
        augmented_X = []
        augmented_y = []
        for text, label in zip(X, y):
            augmented_X.append(text)
            augmented_y.append(label)
            # Apply data augmentation to only the minority class
            if label == minority_class:
                augmented_text = synonym_replacement(text)  # Use other data augmentation techniques as well
                augmented_X.append(augmented_text)
                augmented_y.append(label)

        X = augmented_X
        y = augmented_y

        # Step 2: Feature extraction using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Step 4: Train and evaluate each classifier
        classifiers = {
            'Support Vector Machine': SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Multinomial Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }

        classification_reports = {}
        for name, classifier in classifiers.items():
            print(f"Training and evaluating {name}...")
            report = train_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test)
            classification_reports[name] = report

        # Step 5: Compare the results
        print("\nComparison of Results:")
        for name, report in classification_reports.items():
            print(f"\n{name}:")
            print(pd.DataFrame(report).transpose())

        stdout = io.StringIO()
        # Reset stdout
        sys.stdout = sys.__stdout__  # Reset stdout to its original value

        # Get the captured console output
        console_output = stdout.getvalue()

        # Comparison of Results
        comparison_results = {}
        for name, report in classification_reports.items():
            comparison_results[name] = pd.DataFrame(report).transpose().to_html()
        

        return render_template('ml_results.html', comparison_results=comparison_results)
    return render_template('upload.html')
# Route for user role management
@app.route('/roles')
def roles():
    return render_template('roles.html', users=users)

# Route for integration with external services
@app.route('/integrations')
def integrations():
    return render_template('integrations.html')

# Database of users
users = {
    'user1': {'username': 'user1', 'password': generate_password_hash('password1'), 'role': 'user'},
    'admin1': {'username': 'admin1', 'password': generate_password_hash('adminpassword1'), 'role': 'admin'}
}

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/ml_results')
def ml_results():

    return render_template('ml_results.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        if session['role'] == 'admin':
            return render_template('admin_dashboard.html', username=session['username'])
        else:
            return render_template('user_dashboard.html', username=session['username'])
    else:
        return redirect(url_for('login'))
    
@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        if username not in users:
            users[username] = {'username': username, 'password': generate_password_hash(password), 'role': role}
            return redirect(url_for('dashboard'))
        else:
            return render_template('create_user.html', error='Username already exists')
    return render_template('create_user.html')

if __name__ == '__main__':
    app.run(debug=True)
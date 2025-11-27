# Day 03: Natural Language Processing (NLP) 🤖📝 - Text Preprocessing & Vectorization

This practical session covers fundamental NLP techniques for text preprocessing and vectorization using Python's NLTK and scikit-learn libraries.

## 📁 Files

1. **Text_Preprocessing.ipynb** - Complete text preprocessing pipeline ✨
2. **Text_Preprocessing_1.ipynb** - Basic text cleaning example 🧹
3. **Text_Vectorization.ipynb** - TF-IDF vectorization demonstration 📊

---

## 🔧 1. Text Preprocessing Pipeline

### 📖 Overview

Text preprocessing is essential for preparing raw text data for NLP tasks. This involves cleaning and normalizing text through multiple steps.

### 🛠️ Key Techniques

#### **🔤 Lowercasing**

Converts all text to lowercase for consistency.

```python
text = text.lower()
```

#### **🚫 Remove Punctuation**

Removes special characters and punctuation marks.

```python
text = text.translate(str.maketrans('', '', string.punctuation))
```

#### **✂️ Tokenization**

Splits text into individual words (tokens).

```python
tokens = word_tokenize(text)
```

#### **🗑️ Stop Words Removal**

Removes common words (like "the", "is", "and") that don't carry significant meaning.

```python
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
```

#### **🌱 Stemming**

Reduces words to their root form (e.g., "running" → "run").

```python
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
```

#### **📚 Lemmatization**

Converts words to their base dictionary form (e.g., "better" → "good").

```python
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
```

### 💻 Example Code

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

text = "Hello! This is an Example showing, how text Pre-processing works better."

# Lowercasing
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokens = word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
```

---

## 📊 2. TF-IDF Vectorization

### 📖 Overview

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection of documents.

### 🤔 What is TF-IDF?

- **📈 TF (Term Frequency)**: How often a word appears in a document
- **🔍 IDF (Inverse Document Frequency)**: How unique/rare a word is across all documents
- **⭐ TF-IDF Score**: TF × IDF - balances word frequency with uniqueness

### 💡 Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Define sample documents
documents = [
    "Natural Language Processing is a subfield of artificial intelligence.",
    "It focuses on the interaction between computers and humans using natural language.",
    "TF-IDF is a technique used to evaluate the importance of words in a document."
]

# Initialize and fit the vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert to dense array for viewing
dense_matrix = tfidf_matrix.toarray()

# Display TF-IDF scores for each document
for i, doc in enumerate(dense_matrix):
    print(f"\nDocument {i+1} TF-IDF Scores:")
    for word, score in zip(feature_names, doc):
        print(f"{word}: {score:.4f}")
```

### ⚙️ How It Works

1. **🔢 Vectorization**: Converts text documents into numerical vectors
2. **📦 Sparse Matrix**: Efficiently stores the TF-IDF values
3. **📝 Feature Names**: Unique words (vocabulary) across all documents
4. **🎯 Scoring**: Each word gets a TF-IDF score indicating its importance

---

## 🛠️ Setup & Requirements

### 📦 Install Dependencies

```bash
pip install nltk scikit-learn pandas
```

### ⬇️ Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 💡 Key Takeaways

1. **🧹 Text Preprocessing** prepares raw text for analysis by cleaning and normalizing it
2. **⚖️ Stemming vs Lemmatization**: Stemming is faster but cruder; lemmatization is more accurate
3. **🎯 TF-IDF** helps identify important words by balancing frequency with uniqueness
4. **🔢 Vectorization** converts text into numbers that machine learning algorithms can process

---

## 🚀 Use Cases

- **😊 Sentiment Analysis**: Preprocessing text before analyzing emotions
- **📂 Text Classification**: Categorizing documents by topic
- **🔎 Information Retrieval**: Search engines and document ranking
- **💬 Chatbots & NLP Applications**: Understanding and processing user input

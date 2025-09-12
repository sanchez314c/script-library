import nltk
import spacy
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(filename='skills_extractor.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

ai_skills = [
    'machine learning', 'natural language processing', 'deep learning', 'neural networks',
    'python', 'data science', 'computer vision'
    # Add more skills as needed
]

additional_keywords = [
    'python', 'data science', 'computer vision',
    # Add more keywords as needed
]

def expand_keywords_with_synonyms(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name().replace('_', ' '))
    return list(expanded_keywords)

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_skills(text, skills_list):
    tokens = word_tokenize(text.lower())
    return [skill for skill in skills_list if skill.lower() in tokens]

def detect_anomalies(content):
    anomaly_keywords = ['anomaly', 'glitch', 'unexpected', 'emerging', 'security']
    return any(keyword in content.lower() for keyword in anomaly_keywords)

def process_text(text):
    preprocessed_text = preprocess(text)
    extracted_skills = extract_skills(preprocessed_text, ai_skills)
    entities = perform_ner(preprocessed_text)
    anomalies = detect_anomalies(preprocessed_text)
    
    return {
        'preprocessed_text': preprocessed_text,
        'skills': extracted_skills,
        'entities': entities,
        'anomalies': anomalies
    }

if __name__ == "__main__":
    sample_text = "I have experience in machine learning and natural language processing."
    result = process_text(sample_text)
    print(result)

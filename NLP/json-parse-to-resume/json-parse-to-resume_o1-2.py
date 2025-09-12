import os
import json
import tkinter as tk
from tkinter import filedialog
import logging
import nltk
import spacy
import torch
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertTokenizer, BertModel
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime

# Optional: Suppress warnings from transformers regarding parameter renaming
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*renamed internally.*")

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Set up NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_support")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir)

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load BERT for contextual understanding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Setup logging
logging.basicConfig(filename='conversation_parser.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check if MPS (Apple Metal) is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS (Apple Metal) is available and will be used.")
else:
    mps_device = torch.device("cpu")
    print("MPS (Apple Metal) is NOT available. Using CPU instead.")

# Initialize summarization pipeline and specify the device
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=mps_device)

# Define AI-related skills and additional keywords
ai_skills = [
    "machine learning", "deep learning", "neural networks", "natural language processing",
    "computer vision", "data analysis", "data visualization", "python programming",
    "model training", "hyperparameter tuning", "reinforcement learning", "unsupervised learning",
    "supervised learning", "tensorflow", "pytorch", "scikit-learn", "data preprocessing",
    "algorithm development", "artificial intelligence", "predictive modeling", "big data",
    "cloud computing", "database management", "feature engineering", "statistical analysis",
    "time series analysis", "anomaly detection", "image processing", "text mining",
    "transfer learning", "model evaluation", "optimization techniques"
]

additional_keywords = [
    "project", "task", "assignment", "development", "implementation", "contributed", "led", 
    "developed", "implemented", "designed", "achieved", "completed", "successfully", "solved", 
    "resolved", "used", "utilized", "employed", "applied", "course", "certification", "self-taught", 
    "education", "training", "teamwork", "communication", "leadership", "problem-solving", 
    "adaptability", "challenge", "difficulty", "problem", "obstacle", "issue", "recommended", 
    "positive feedback", "commendation", "appreciation", "idea", "innovation", "proposed", 
    "suggested", "methodology", "process", "framework", "approach"
]

def expand_keywords_with_synonyms(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name().replace('_', ' '))
    return list(expanded_keywords)

ai_skills = expand_keywords_with_synonyms(ai_skills)
additional_keywords = expand_keywords_with_synonyms(additional_keywords)

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def perform_topic_modeling(texts, num_topics=5, num_words=10):
    # Ensure valid max_df, min_df configuration
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")
    return topics

def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def cluster_skills(skills, num_clusters=5):
    if not skills:
        return {}
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(skills)
    km = KMeans(n_clusters=min(num_clusters, len(skills)), random_state=42)  # Avoid error if fewer skills than clusters
    km.fit(X)
    clusters = defaultdict(list)
    for idx, label in enumerate(km.labels_):
        clusters[label].append(skills[idx])
    return clusters

def correlate_keywords(messages, keywords):
    correlated_entries = {keyword: [] for keyword in keywords}
    for msg in messages:
        text = " ".join(msg['content'])
        preprocessed_text = preprocess(text)
        for keyword in keywords:
            if keyword in preprocessed_text:
                correlated_entries[keyword].append({
                    'text': text,
                    'timestamp': msg['timestamp']
                })
    return correlated_entries

def generate_resume_entries(correlated_entries, keywords):
    entries = []
    for keyword, details in correlated_entries.items():
        if details:
            entry = {
                'keyword': keyword,
                'details': details
            }
            entries.append(entry)
    return entries

def split_text(text, max_length):
    words = text.split()
    chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
    return chunks

def summarize_messages(mapping, current_id):
    summaries = []
    if current_id in mapping:
        entry = mapping[current_id]
        message = entry.get('message', {})
        if message:
            content = " ".join(message.get('content', {}).get('parts', []))
            preprocessed_text = preprocess(content)
            text_chunks = split_text(preprocessed_text, 500)  # Split into chunks of max 500 words
            summarized_chunks = []
            for chunk in text_chunks:
                chunk_length = len(chunk.split())
                max_len = min(chunk_length, 150)
                min_len = min(30, max_len - 1)
                
                summarized_text = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                summarized_chunks.append(summarized_text)
            summary = ' '.join(summarized_chunks)
            summaries.append({
                'content': content,
                'summary': summary,
                'timestamp': entry.get('create_time'),
            })
        for child_id in entry.get('children', []):
            summaries.extend(summarize_messages(mapping, child_id))
    return summaries

def extract_messages(mapping, current_id):
    messages = []
    if current_id in mapping:
        entry = mapping[current_id]
        message = entry.get('message', {})
        if message:
            content = message.get('content', {}).get('parts', [])
            messages.append({
                'author_role': message.get('author', {}).get('role', 'unknown'),
                'content': content,
                'timestamp': entry.get('create_time'),
            })
        for child_id in entry.get('children', []):
            messages.extend(extract_messages(mapping, child_id))
    return messages

def process_conversations(input_path, skills, additional_keywords):
    with open(input_path, 'r') as file:
        data = json.load(file)
    processed_conversations = []
    for conversation in data:
        conversation_id = conversation.get('id')
        title = conversation.get('title', 'Untitled')
        create_time = conversation.get('create_time')
        update_time = conversation.get('update_time')
        mapping = conversation.get('mapping', {})
        root_id = conversation.get('current_node')
        
        # Summarize conversations
        summaries = summarize_messages(mapping, root_id)
        
        # Perform NER, sentiment analysis, and topic modeling on summaries
        for summary in summaries:
            summary['entities'] = perform_ner(summary['content'])
            summary['sentiment'] = perform_sentiment_analysis(summary['content'])
        
        all_content = [summary['content'] for summary in summaries]
        topics = perform_topic_modeling(all_content) if all_content else []
        
        # Extract messages
        messages = extract_messages(mapping, root_id)
        
        # Correlate keywords
        correlated_skills = correlate_keywords(messages, skills)
        correlated_additional = correlate_keywords(messages, additional_keywords)
        
        # Generate resume entries
        resume_entries_skills = generate_resume_entries(correlated_skills, skills)
        resume_entries_additional = generate_resume_entries(correlated_additional, additional_keywords)
        
        # Cluster identified skills
        skill_keywords = [entry['keyword'] for entry in resume_entries_skills]
        skill_clusters = cluster_skills(skill_keywords)

        processed_conversations.append({
            'id': conversation_id,
            'title': title,
            'create_time': create_time,
            'update_time': update_time,
            'summaries': summaries,
            'topics': topics,
            'resume_entries_skills': resume_entries_skills,
            'resume_entries_additional': resume_entries_additional,
            'skill_clusters': skill_clusters
        })
    return processed_conversations

def perform_temporal_analysis(processed_conversations):
    skill_timeline = defaultdict(lambda: defaultdict(int))
    
    for conversation in processed_conversations:
        if conversation['create_time']:
            conversation_date = datetime.fromtimestamp(conversation['create_time']).date()
            for entry in conversation['resume_entries_skills']:
                skill_timeline[entry['keyword']][conversation_date] += 1

    return skill_timeline

def plot_skill_timeline(skill_timeline):
    if not skill_timeline:
        return
    plt.figure(figsize=(15, 10))
    for skill, timeline in skill_timeline.items():
        dates = sorted(timeline.keys())
        if dates:
            counts = [timeline[date] for date in dates]
            plt.plot(dates, counts, label=skill)
    
    plt.xlabel('Date')
    plt.ylabel('Skill Mention Count')
    plt.title('Skill Development Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('skill_timeline.png')
    plt.close()

def compare_with_job_description(your_skills, job_description):
    your_skills_text = ' '.join(your_skills)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([your_skills_text, job_description])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    job_desc_words = set(job_description.lower().split())
    your_skills_lower = set(skill.lower() for skill in your_skills)
    
    matching_skills = your_skills_lower.intersection(job_desc_words)
    missing_skills = job_desc_words - your_skills_lower
    
    return {
        'similarity_score': similarity,
        'matching_skills': list(matching_skills),
        'missing_skills': list(missing_skills)
    }

def save_summaries(processed_conversations, output_file):
    with open(output_file, 'w') as file:
        for convo in processed_conversations:
            file.write(f"Conversation ID: {convo['id']}\n")
            file.write(f"Title: {convo['title']}\n")
            file.write(f"Created: {convo['create_time']}\n")
            file.write(f"Updated: {convo['update_time']}\n\n")
            for summary in convo['summaries']:
                file.write(f"Timestamp: {summary['timestamp']}\n")
                file.write(f"Content: {summary['content']}\n")
                file.write(f"Summary: {summary['summary']}\n")
                file.write(f"Entities: {summary['entities']}\n")
                file.write(f"Sentiment: {summary['sentiment']}\n\n")
            file.write(f"Topics: {convo['topics']}\n")
            file.write("\n" + "-"*50 + "\n\n")

def write_skills_to_file(skills, keywords, output_file):
    with open(output_file, 'w') as file:
        file.write("AI Skills:\n")
        for skill in sorted(skills):
            file.write(f"{skill}\n")
        file.write("\nAdditional Keywords:\n")
        for keyword in sorted(keywords):
            file.write(f"{keyword}\n")

def generate_supporting_text(conversations, skills, keywords):
    supporting_text = {skill: [] for skill in skills + keywords}
    
    for conversation in conversations:
        for entry in conversation.get('resume_entries_skills', []):
            keyword = entry['keyword']
            if keyword in supporting_text:
                for detail in entry['details']:
                    supporting_text[keyword].append(detail['text'])
        
        for entry in conversation.get('resume_entries_additional', []):
            keyword = entry['keyword']
            if keyword in supporting_text:
                for detail in entry['details']:
                    supporting_text[keyword].append(detail['text'])
    
    return supporting_text

def save_supporting_text(supporting_text, output_file):
    with open(output_file, 'w') as file:
        for keyword, texts in supporting_text.items():
            file.write(f"{keyword}:\n")
            for text in texts:
                file.write(f"  - {text}\n")
            file.write("\n")

def save_plain_text_overview(processed_conversations, output_file):
    with open(output_file, 'w') as file:
        for convo in processed_conversations:
            file.write(f"Conversation ID: {convo['id']}\n")
            file.write(f"Title: {convo['title']}\n")
            file.write(f"Created: {convo['create_time']}\n")
            file.write(f"Updated: {convo['update_time']}\n\n")
            
            file.write("Summaries:\n")
            for summary in convo['summaries']:
                file.write(f"Timestamp: {summary['timestamp']}\n")
                file.write(f"Summary: {summary['summary']}\n\n")
            
            file.write("AI Skills Detected:\n")
            for entry in convo['resume_entries_skills']:
                file.write(f"Skill: {entry['keyword']}\n")
                for detail in entry['details']:
                    file.write(f"  - {detail['text']}\n")
            
            file.write("\nAdditional Keywords Detected:\n")
            for entry in convo['resume_entries_additional']:
                file.write(f"Keyword: {entry['keyword']}\n")
                for detail in entry['details']:
                    file.write(f"  - {detail['text']}\n")
            
            file.write("\n" + "="*50 + "\n\n")

def main():
    try:
        # Prompt the user to select the source directory for JSON files
        input_directory = filedialog.askdirectory(
            title="Select the JSON source directory"
        )
        if not input_directory:
            print("No input directory selected. Exiting...")
            logging.info("No input directory selected by user.")
            return

        # Prompt the user to select the output folder
        output_folder = filedialog.askdirectory(
            title="Select the output folder"
        )
        if not output_folder:
            print("No output folder selected. Exiting...")
            logging.info("No output folder selected by user.")
            return

        # Process all JSON files recursively in the source directory
        processed_conversations = process_all_json_files_in_directory(input_directory, ai_skills, additional_keywords)

        # Save summaries
        summary_output_file = os.path.join(output_folder, 'conversation_summaries.txt')
        save_summaries(processed_conversations, summary_output_file)
        
        # Extract skills and keywords
        skills_set = set()
        keywords_set = set()
        for convo in processed_conversations:
            for entry in convo['resume_entries_skills']:
                skills_set.add(entry['keyword'])
            for entry in convo['resume_entries_additional']:
                keywords_set.add(entry['keyword'])

        # Write the skills and keywords to a text file
        skills_output_file = os.path.join(output_folder, 'skills.txt')
        write_skills_to_file(skills_set, keywords_set, skills_output_file)

        # Generate and save supporting text
        supporting_text = generate_supporting_text(processed_conversations, ai_skills, additional_keywords)
        supporting_text_file = os.path.join(output_folder, 'supporting_text.txt')
        save_supporting_text(supporting_text, supporting_text_file)

        # Output the resume entries to a JSON file
        resume_entries_file = os.path.join(output_folder, 'resume_entries.json')
        with open(resume_entries_file, 'w') as outfile:
            json.dump(processed_conversations, outfile, indent=4)

        # Save a plain text overview of all results
        plain_text_overview_file = os.path.join(output_folder, 'plain_text_overview.txt')
        save_plain_text_overview(processed_conversations, plain_text_overview_file)

        print(f"Summarized conversations have been saved to '{summary_output_file}'")
        print(f"Skills and keywords have been saved to '{skills_output_file}'")
        print(f"Supporting text has been saved to '{supporting_text_file}'")
        print(f"Resume entries have been saved to '{resume_entries_file}'")
        print(f"Plain text overview has been saved to '{plain_text_overview_file}'")

        # Perform temporal analysis
        skill_timeline = perform_temporal_analysis(processed_conversations)
        plot_skill_timeline(skill_timeline)

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

import json
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import logging
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Set up the directory for NLTK support files
script_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(script_dir, "ntlk_support")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

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

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

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
            text_chunks = split_text(preprocessed_text, 500)
            summarized_chunks = []
            for chunk in text_chunks:
                chunk_length = len(chunk.split())
                # Ensure max_len >= min_len and min_len < max_len
                max_len = min(chunk_length, 150)
                # If chunk_length < 30, set min_len to a smaller number
                # min_len must be at least 1 and less than max_len
                if max_len < 2:
                    # Very short text: max_len=1 or max_len=0 means no summarization needed
                    # Just append the chunk as is, or skip if empty
                    if max_len == 1:
                        summarized_chunks.append(chunk)
                    # If there's literally no words, skip
                    continue
                min_len = min(30, max_len - 1)
                
                # Handle the case where chunk_length is really small
                if min_len >= max_len:
                    # If text is too short, just treat it as is, no summarization needed
                    summarized_chunks.append(chunk)
                else:
                    summarized = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                    summarized_chunks.append(summarized)
            
            summary = ' '.join(summarized_chunks) if summarized_chunks else content
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
        
        summaries = summarize_messages(mapping, root_id)
        messages = extract_messages(mapping, root_id)
        correlated_skills = correlate_keywords(messages, skills)
        correlated_additional = correlate_keywords(messages, additional_keywords)
        resume_entries_skills = generate_resume_entries(correlated_skills, skills)
        resume_entries_additional = generate_resume_entries(correlated_additional, additional_keywords)
        
        processed_conversations.append({
            'id': conversation_id,
            'title': title,
            'create_time': create_time,
            'update_time': update_time,
            'summaries': summaries,
            'resume_entries_skills': resume_entries_skills,
            'resume_entries_additional': resume_entries_additional
        })
    return processed_conversations

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
                file.write(f"Summary: {summary['summary']}\n\n")
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

def process_all_json_files_in_directory(source_directory, skills, additional_keywords):
    processed_conversations = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                processed_conversations.extend(process_conversations(file_path, skills, additional_keywords))
    return processed_conversations

def main():
    root = tk.Tk()
    root.withdraw()

    try:
        input_directory = filedialog.askdirectory(title="Select the JSON source directory")
        if not input_directory:
            print("No input directory selected. Exiting...")
            logging.info("No input directory selected by user.")
            return

        output_folder = filedialog.askdirectory(title="Select the output folder")
        if not output_folder:
            print("No output folder selected. Exiting...")
            logging.info("No output folder selected by user.")
            return

        processed_conversations = process_all_json_files_in_directory(input_directory, ai_skills, additional_keywords)

        summary_output_file = os.path.join(output_folder, 'conversation_summaries.txt')
        save_summaries(processed_conversations, summary_output_file)
        
        skills_set = set()
        keywords_set = set()
        for convo in processed_conversations:
            for entry in convo['resume_entries_skills']:
                skills_set.add(entry['keyword'])
            for entry in convo['resume_entries_additional']:
                keywords_set.add(entry['keyword'])

        skills_output_file = os.path.join(output_folder, 'skills.txt')
        write_skills_to_file(skills_set, keywords_set, skills_output_file)

        supporting_text = generate_supporting_text(processed_conversations, ai_skills, additional_keywords)
        supporting_text_file = os.path.join(output_folder, 'supporting_text.txt')
        save_supporting_text(supporting_text, supporting_text_file)

        resume_entries_file = os.path.join(output_folder, 'resume_entries.json')
        with open(resume_entries_file, 'w') as outfile:
            json.dump(processed_conversations, outfile, indent=4)

        plain_text_overview_file = os.path.join(output_folder, 'plain_text_overview.txt')
        save_plain_text_overview(processed_conversations, plain_text_overview_file)

        print(f"Summarized conversations have been saved to '{summary_output_file}'")
        print(f"Skills and keywords have been saved to '{skills_output_file}'")
        print(f"Supporting text has been saved to '{supporting_text_file}'")
        print(f"Resume entries have been saved to '{resume_entries_file}'")
        print(f"Plain text overview has been saved to '{plain_text_overview_file}'")

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

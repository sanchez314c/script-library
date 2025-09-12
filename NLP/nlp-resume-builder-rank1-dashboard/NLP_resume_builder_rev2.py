import logging
import os
import glob
import json
import spacy
import tkinter as tk
from tkinter import filedialog
from collections import Counter, defaultdict
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jsonschema
import networkx as nx
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure MPS is used for torch if available, fallback to CUDA or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
logging.info(f"Using device: {device}")

# Load the sentence-transformers model with MPS if available
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load BERT tokenizer and model for sentiment analysis with MPS
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=0 if device == "cuda" else -1)

# Load SpaCy Transformer Model (optimized for accuracy)
nlp = spacy.load("en_core_web_trf")

# Define JSON schema for validation
json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "user_input": {"type": "string"},
            "ai_response": {"type": "string"},
        },
        "required": ["timestamp", "user_input", "ai_response"],
    },
}

def validate_json(data):
    try:
        for entry in data:
            # Ensure the required 'timestamp', 'user_input', and 'ai_response' fields are present
            if 'timestamp' not in entry:
                entry['timestamp'] = 'Unknown'  # You can replace 'Unknown' with a default value
            if 'user_input' not in entry:
                entry['user_input'] = 'No user input provided'  # Default text if missing
            if 'ai_response' not in entry:
                entry['ai_response'] = 'No AI response available'  # Default text if missing

        # Proceed with schema validation after fixing missing fields
        jsonschema.validate(instance=data, schema=json_schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f"JSON validation error: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        return False

def load_and_clean_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.json"))
    data = []
    for file in all_files:
        try:
            with open(file, "r") as f:
                file_data = json.load(f)
                if validate_json(file_data):
                    data.extend(file_data)
                else:
                    logging.warning(f"Skipping invalid JSON file: {file}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in file {file}: {e}")
            # Attempt to load the file line by line to identify the problematic part
            try:
                with open(file, "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as line_error:
                            logging.error(f"Error decoding JSON on line {i+1} of file {file}: {line_error}")
            except Exception as e:
                logging.error(f"Unexpected error reading file {file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error reading file {file}: {e}")
    return data

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

def preprocess_all_texts(data):
    all_texts = [item['user_input'] for item in data] + [item['ai_response'] for item in data]
    
    with ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor here
        results = list(executor.map(preprocess_text, all_texts))

    return results

def process_text(doc):
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    embedding_value = None

    try:
        embedding = model.encode(" ".join(tokens), convert_to_tensor=True)
        embedding_value = embedding.mean().item()  # Calculate mean directly

    except Exception as e:
        logging.error(f"Error processing embeddings: {e}")
        embedding_value = None

    return tokens, entities, dependencies, embedding_value

def extract_entities_and_tokens_parallel(preprocessed_data):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_text, nlp.pipe(preprocessed_data, batch_size=100)))

    tokens, entities, dependencies, embedding_values = zip(*results)

    logging.info(f"Tokens: {tokens[:2]}")
    logging.info(f"Entities: {entities[:2]}")
    logging.info(f"Dependencies: {dependencies[:2]}")
    logging.info(f"Embedding Values: {embedding_values[:2]}")

    return tokens, entities, dependencies, embedding_values

def get_top_skills(tokens):
    all_tokens = [token for sublist in tokens for token in sublist]
    skill_counts = Counter(all_tokens)
    stop_words = set(["the", "a", "an", "in", "to", "of", "and", "for", "is", "are", "be", "with", "that", "this", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "not", "it", "i", "you", "he", "she", "they", "we", "us", "them", "our", "your", "their", "my", "me", "myself", "yourself", "himself", "herself", "itself", "themself", "ourselves", "yourselves", "themselves", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
    filtered_skills = [(skill, count) for skill, count in skill_counts.items() if skill not in stop_words and len(skill) > 2]
    top_skills = sorted(filtered_skills, key=lambda item: item[1], reverse=True)
    return top_skills

# Store embeddings in a dictionary
embedding_cache = {}

def get_semantic_embedding(text, reduce_to_scalar=True):
    if text in embedding_cache:
        embedding = embedding_cache[text]
    else:
        embedding = model.encode(text, convert_to_tensor=True)
        embedding_cache[text] = embedding

    if reduce_to_scalar:
        if embedding.numel() == 1:
            return embedding.item()
        else:
            return embedding.mean().item()
    else:
        return embedding

def match_skills_to_jobs(skills, job_descriptions):
    skill_text = ' '.join([skill[0] for skill in skills])
    skill_embedding = get_semantic_embedding(skill_text)
    job_embeddings = [get_semantic_embedding(job) for job in job_descriptions]
    
    cosine_similarities = util.pytorch_cos_sim(skill_embedding, job_embeddings).flatten()
    job_matches = sorted(list(zip(job_descriptions, cosine_similarities)), key=lambda x: x[1], reverse=True)
    return job_matches

def extract_timeline(data):
    timeline = []
    for item in data:
        try:
            timestamp = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logging.warning(f"Invalid timestamp format: {item['timestamp']}. Skipping.")
            continue  # Skip this item if the timestamp is invalid

        event = f"Conversation about: {item['user_input'][:50]}"
        timeline.append((timestamp.strftime('%Y-%m-%d'), event))
    timeline.sort(key=lambda x: x[0])
    return timeline

def cluster_and_visualize_skills(skills):
    G = nx.Graph()
    for skill, count in skills:
        G.add_node(skill, size=count)
    
    for i, (skill1, _) in enumerate(skills):
        for skill2, _ in skills[i+1:]:
            if fuzz.ratio(skill1, skill2) > 70:
                G.add_edge(skill1, skill2)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=[G.nodes[node]['size']*10 for node in G.nodes()])
    plt.title("Skill Cluster Visualization")
    plt.savefig("skill_cluster.png")
    plt.close()

def perform_topic_modeling(preprocessed_data):
    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topics, _ = topic_model.fit_transform(preprocessed_data)
    return topic_model.get_topic_info(), topics

def analyze_skill_evolution(data, skills):
    skill_evolution = defaultdict(lambda: defaultdict(int))
    for item, skill_list in zip(data, skills):
        date = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m')
        for skill in skill_list:
            skill_evolution[date][skill] += 1
    
    df = pd.DataFrame(skill_evolution).T.fillna(0)
    df.plot(figsize=(15, 10))
    plt.title("Skill Evolution Over Time")
    plt.savefig("skill_evolution.png")
    plt.close()
    return df

def analyze_sentiment(data):
    sentiments = []
    for item in data:
        sentiment = sentiment_analyzer(item['user_input'])[0]
        sentiments.append((item['timestamp'], sentiment['label'], sentiment['score']))
    return sentiments

def generate_enhanced_resume(name, job_matches, skills, timeline, topics, skill_cluster_image):
    doc = SimpleDocTemplate(f"{name}_AI_generated_resume.pdf", pagesize=letter)
    styles = getSampleStyleSheet() 
    story = []

    story.append(Paragraph(f"{name}'s AI-Generated Resume", styles['Title']))
    story.append(Spacer(1, 12))

    # Job Matches
    story.append(Paragraph("Top Job Matches", styles['Heading2']))
    job_data = [[job, f"{score:.2f}"] for job, score in job_matches[:5]] 
    job_table = Table(job_data, colWidths=[400, 100])
    job_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(job_table)
    story.append(Spacer(1, 12))

    # Skills
    story.append(Paragraph("Key Skills", styles['Heading2']))
    skill_data = [[skill, count] for skill, count in skills[:15]]
    skill_table = Table(skill_data, colWidths=[400, 100])
    skill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(skill_table)
    story.append(Spacer(1, 12))

    # Timeline
    story.append(Paragraph("AI Journey Timeline", styles['Heading2']))
    timeline_data = [[date, event] for date, event in timeline[:10]]
    timeline_table = Table(timeline_data, colWidths=[100, 400])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(timeline_table)
    story.append(Spacer(1, 12))

    # Topics
    story.append(Paragraph("Key Topics", styles['Heading2']))
    topic_data = [[f"Topic {row['Topic']}", row['Name'], row['Count']] for _, row in topics.iterrows() if row['Topic'] != -1][:10]
    topic_table = Table(topic_data, colWidths=[100, 300, 100])
    topic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(topic_table)
    story.append(Spacer(1, 12))

    # Add skill cluster visualization
    story.append(Paragraph("Skill Cluster Visualization", styles['Heading2']))
    story.append(Image(skill_cluster_image, width=500, height=300))

    doc.build(story)
    logging.info(f"Enhanced resume saved as: {name}_AI_generated_resume.pdf")

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing JSON Files")
    return folder_path

def create_dashboard(skills, job_matches, skill_evolution_df, topic_info, conversation_sentiments):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("AI Career Analysis Dashboard"),
        dcc.Tabs([
            dcc.Tab(label='Skills', children=[
                dcc.Graph(id='skill-graph', figure=px.bar(skills[:20], x=0, y=1, title="Top 20 Skills"))
            ]),
            dcc.Tab(label='Job Matches', children=[
                dcc.Graph(id='job-match-graph', figure=px.bar(job_matches, x=0, y=1, title="Job Match Scores"))
            ]),
            dcc.Tab(label='Skill Evolution', children=[
                dcc.Graph(id='skill-evolution-graph', figure=px.line(skill_evolution_df))
            ]),
            dcc.Tab(label='Topics', children=[
                dcc.Graph(id='topic-graph', figure=px.scatter(topic_info, x='Count', y='Name', title="Topic Distribution"))
            ]),
            dcc.Tab(label='Sentiment', children=[
                dcc.Graph(id='sentiment-graph', figure=px.scatter(conversation_sentiments, x=0, y=2, color=1, title="Conversation Sentiment Over Time"))
            ])
        ])
    ])

    return app

def main():
    try:
        folder_path = select_folder()
        if not folder_path:
            logging.error("No folder selected. Exiting.")
            return

        logging.info(f"Loading data from: {folder_path}")
        
        data = load_and_clean_data(folder_path)
        
        if not data:
            logging.error("No valid data found in the selected folder. Exiting.")
            return
        
        preprocessed_data = preprocess_all_texts(data)
        
        tokens, entities, dependencies, _ = extract_entities_and_tokens_parallel(preprocessed_data)
        
        skills = get_top_skills(tokens)
        
        job_descriptions = [
            "AI Engineer with experience in machine learning and deep learning",
            "Data Scientist proficient in statistical analysis and predictive modeling",
            "NLP Specialist with expertise in text processing and language models",
            "Computer Vision Engineer skilled in image processing and object detection",
            "Robotics Engineer with knowledge of control systems and sensor fusion",
            "AI Ethics Researcher focusing on fairness and transparency in AI systems",
            "AI Product Manager with a strong technical background in AI and ML",
            "AI Research Scientist pushing the boundaries of artificial general intelligence",
            "MLOps Engineer specializing in deploying and scaling ML models in production",
            "AI in Healthcare Specialist applying AI to medical diagnosis and treatment"
        ]
        
        job_matches = match_skills_to_jobs(skills, job_descriptions)
        
        timeline = extract_timeline(data)
        
        cluster_and_visualize_skills(skills[:50])
        
        topic_info, document_topics = perform_topic_modeling(preprocessed_data)
        
        skill_evolution_df = analyze_skill_evolution(data, tokens)
        
        conversation_sentiments = analyze_sentiment(data)
        
        name = input("Enter your name: ")
        generate_enhanced_resume(name, job_matches, skills, timeline, topic_info, "skill_cluster.png")
        
        app = create_dashboard(skills, job_matches, skill_evolution_df, topic_info, conversation_sentiments)
        app.run_server(debug=True)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

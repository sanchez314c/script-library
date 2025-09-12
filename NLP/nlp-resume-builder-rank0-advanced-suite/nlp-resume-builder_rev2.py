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
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure MPS is used for torch if available, fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
logging.info(f"Using device: {device}")

try:
    # Load the sentence-transformers model with MPS if available
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Load BERT tokenizer and model for sentiment analysis
    sentiment_analyzer = pipeline('sentiment-analysis', 
                                model='distilbert-base-uncased-finetuned-sst-2-english', 
                                device=device)
    
    # Load SpaCy Transformer Model (optimized for accuracy)
    nlp = spacy.load("en_core_web_trf")
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise

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
    if not isinstance(data, list):
        logging.error("Data must be a list of conversation entries")
        return False
    
    try:
        for entry in data:
            if not isinstance(entry, dict):
                logging.error(f"Invalid entry format: {entry}")
                continue
                
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'user_input' not in entry:
                entry['user_input'] = 'No user input provided'
            if 'ai_response' not in entry:
                entry['ai_response'] = 'No AI response available'

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
    if not all_files:
        logging.error(f"No JSON files found in {folder_path}")
        return []
        
    data = []
    for file in all_files:
        try:
            with open(file, "r", encoding='utf-8') as f:
                file_data = json.load(f)
                if validate_json(file_data):
                    data.extend(file_data)
                else:
                    logging.warning(f"Skipping invalid JSON file: {file}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in file {file}: {e}")
            try:
                with open(file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as line_error:
                            logging.error(f"Error on line {i+1}: {line_error}")
            except Exception as e:
                logging.error(f"Error reading file {file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error reading file {file}: {e}")
    
    if not data:
        logging.warning("No valid data loaded from any files")
    return data

def preprocess_text(text):
    try:
        doc = nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ""

def preprocess_all_texts(data):
    try:
        all_texts = [item['user_input'] for item in data] + [item['ai_response'] for item in data]
        if not all_texts:
            logging.error("No texts to process")
            return []
            
        max_workers = min(os.cpu_count() or 1, len(all_texts))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(preprocess_text, all_texts))
        
        return [r for r in results if r]  # Filter out empty results
    except Exception as e:
        logging.error(f"Error in parallel text processing: {e}")
        return []

def process_text(doc):
    try:
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

        try:
            text = " ".join(tokens)
            if not text.strip():
                return [], [], [], 0.0
                
            embedding = model.encode(text, convert_to_tensor=True)
            embedding_value = embedding.mean().item() if embedding is not None else 0.0

        except Exception as e:
            logging.error(f"Error processing embeddings: {e}")
            embedding_value = 0.0

        return tokens, entities, dependencies, embedding_value
    except Exception as e:
        logging.error(f"Error in process_text: {e}")
        return [], [], [], 0.0

def extract_entities_and_tokens_parallel(preprocessed_data):
    if not preprocessed_data:
        logging.error("No preprocessed data to analyze")
        return [], [], [], []
        
    try:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            results = list(executor.map(process_text, nlp.pipe(preprocessed_data, batch_size=100)))

        tokens, entities, dependencies, embedding_values = zip(*results)

        logging.info(f"Tokens sample: {tokens[:2]}")
        logging.info(f"Entities sample: {entities[:2]}")
        logging.info(f"Dependencies sample: {dependencies[:2]}")
        logging.info(f"Embedding Values sample: {embedding_values[:2]}")

        return tokens, entities, dependencies, embedding_values
    except Exception as e:
        logging.error(f"Error in parallel entity extraction: {e}")
        return [], [], [], []

def get_top_skills(tokens):
    if not tokens:
        logging.error("No tokens provided for skill extraction")
        return []
        
    try:
        all_tokens = [token for sublist in tokens for token in sublist]
        skill_counts = Counter(all_tokens)
        
        stop_words = set(["the", "a", "an", "in", "to", "of", "and", "for", "is", "are", "be", "with", 
                         "that", "this", "have", "has", "had", "do", "does", "did", "can", "could", "will", 
                         "would", "should", "may", "might", "must", "not", "it", "i", "you", "he", "she", 
                         "they", "we", "us", "them", "our", "your", "their", "my", "me", "myself", "yourself", 
                         "himself", "herself", "itself", "themself", "ourselves", "yourselves", "themselves", 
                         "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", 
                         "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", 
                         "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
                         "should", "now"])
                         
        filtered_skills = [(skill, count) for skill, count in skill_counts.items() 
                          if skill not in stop_words and len(skill) > 2]
        return sorted(filtered_skills, key=lambda item: item[1], reverse=True)
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        return []

# Store embeddings in a dictionary
embedding_cache = {}

def get_semantic_embedding(text, reduce_to_scalar=True):
    if not text:
        logging.error("Empty text provided for embedding")
        return 0.0 if reduce_to_scalar else torch.zeros(384, device=device)
        
    try:
        if text in embedding_cache:
            embedding = embedding_cache[text]
        else:
            embedding = model.encode(text, convert_to_tensor=True)
            embedding_cache[text] = embedding

        if reduce_to_scalar:
            if embedding.dim() > 1:
                return embedding.mean().item()
            return embedding.item()
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return 0.0 if reduce_to_scalar else torch.zeros(384, device=device)

def match_skills_to_jobs(skills, job_descriptions):
    if not skills or not job_descriptions:
        logging.error("Missing skills or job descriptions")
        return []
        
    try:
        skill_text = ' '.join([skill[0] for skill in skills])
        if not skill_text.strip():
            logging.error("Empty skill text generated")
            return []
            
        skill_embedding = get_semantic_embedding(skill_text, reduce_to_scalar=False)
        job_embeddings = [get_semantic_embedding(job, reduce_to_scalar=False) for job in job_descriptions]
        
        if skill_embedding is None or not job_embeddings:
            logging.error("Failed to generate embeddings")
            return []
            
        cosine_similarities = util.pytorch_cos_sim(skill_embedding, torch.stack(job_embeddings))
        job_matches = sorted(list(zip(job_descriptions, cosine_similarities[0].tolist())), 
                           key=lambda x: x[1], reverse=True)
        return job_matches
    except Exception as e:
        logging.error(f"Error matching skills to jobs: {e}")
        return []

def extract_timeline(data):
    if not data:
        logging.error("No data provided for timeline extraction")
        return []
        
    timeline = []
    for item in data:
        try:
            timestamp = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
            event = f"Conversation about: {item['user_input'][:50]}..."
            timeline.append((timestamp.strftime('%Y-%m-%d'), event))
        except (ValueError, KeyError) as e:
            logging.warning(f"Invalid timestamp format or missing data: {e}")
            continue
    
    timeline.sort(key=lambda x: x[0])
    return timeline

def cluster_and_visualize_skills(skills):
    if not skills:
        logging.error("No skills provided for visualization")
        return
        
    try:
        plt.clf()  # Clear any existing plots
        G = nx.Graph()
        
        # Add nodes with skill counts
        for skill, count in skills:
            G.add_node(skill, size=count)
        
        # Add edges between similar skills
        for i, (skill1, _) in enumerate(skills):
            for skill2, _ in skills[i+1:]:
                if fuzz.ratio(skill1, skill2) > 70:
                    G.add_edge(skill1, skill2)
        
        if not G.nodes():
            logging.error("No nodes in graph")
            return
            
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, 
                node_size=[G.nodes[node]['size']*10 for node in G.nodes()],
                node_color='lightblue',
                font_size=8)
        plt.title("Skill Cluster Visualization")
        plt.savefig("skill_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Error in skill visualization: {e}")

def perform_topic_modeling(preprocessed_data):
    if not preprocessed_data:
        logging.error("No data provided for topic modeling")
        return pd.DataFrame(), []
        
    try:
        topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
        topics, _ = topic_model.fit_transform(preprocessed_data)
        return topic_model.get_topic_info(), topics
    except Exception as e:
        logging.error(f"Error in topic modeling: {e}")
        return pd.DataFrame(), []

def analyze_skill_evolution(data, skills):
    if not data or not skills:
        logging.error("Missing data for skill evolution analysis")
        return pd.DataFrame()
        
    try:
        skill_evolution = defaultdict(lambda: defaultdict(int))
        for item, skill_list in zip(data, skills):
            try:
                date = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m')
                for skill in skill_list:
                    skill_evolution[date][skill] += 1
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing entry: {e}")
                continue
        
        df = pd.DataFrame(skill_evolution).T.fillna(0)
        
        plt.clf()
        df.plot(figsize=(15, 10))
        plt.title("Skill Evolution Over Time")
        plt.savefig("skill_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    except Exception as e:
        logging.error(f"Error in skill evolution analysis: {e}")
        return pd.DataFrame()

def analyze_sentiment(data):
    if not data:
        logging.error("No data provided for sentiment analysis")
        return []
        
    sentiments = []
    for item in data:
        try:
            sentiment = sentiment_analyzer(item['user_input'])[0]
            sentiments.append((item['timestamp'], sentiment['label'], sentiment['score']))
        except Exception as e:
            logging.warning(f"Error analyzing sentiment: {e}")
            continue
    return sentiments

def generate_enhanced_resume(name, job_matches, skills, timeline, topics, skill_cluster_image):
    try:
        doc = SimpleDocTemplate(f"{name}_AI_generated_resume.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph(f"{name}'s AI-Generated Resume", styles['Title']))
        story.append(Spacer(1, 12))

        # Add sections only if data is available
        if job_matches:
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

        # Add other sections similarly with checks...
        # (Previous table styling code remains the same)

        if os.path.exists(skill_cluster_image):
            story.append(Paragraph("Skill Cluster Visualization", styles['Heading2']))
            story.append(Image(skill_cluster_image, width=500, height=300))

        doc.build(story)
        logging.info(f"Enhanced resume saved as: {name}_AI_generated_resume.pdf")
    except Exception as e:
        logging.error(f"Error generating resume: {e}")

def create_dashboard(skills, job_matches, skill_evolution_df, topic_info, conversation_sentiments):
    if not all([skills, job_matches, skill_evolution_df is not None, 
                topic_info is not None, conversation_sentiments]):
        logging.error("Insufficient data for dashboard creation")
        return None

    try:
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("AI Career Analysis Dashboard"),
            dcc.Tabs([
                dcc.Tab(label='Skills', children=[
                    dcc.Graph(id='skill-graph', 
                             figure=px.bar(skills[:20], x=0, y=1, 
                                         title="Top 20 Skills"))
                ]),
                dcc.Tab(label='Job Matches', children=[
                    dcc.Graph(id='job-match-graph', 
                             figure=px.bar(job_matches, x=0, y=1, 
                                         title="Job Match Scores"))
                ]),
                dcc.Tab(label='Skill Evolution', children=[
                    dcc.Graph(id='skill-evolution-graph', 
                             figure=px.line(skill_evolution_df))
                ]),
                dcc.Tab(label='Topics', children=[
                    dcc.Graph(id='topic-graph', 
                             figure=px.scatter(topic_info, x='Count', y='Name', 
                                             title="Topic Distribution"))
                ]),
                dcc.Tab(label='Sentiment', children=[
                    dcc.Graph(id='sentiment-graph', 
                             figure=px.scatter(conversation_sentiments, 
                                             x=0, y=2, color=1, 
                                             title="Conversation Sentiment Over Time"))
                ])
            ])
        ])
        return app
    except Exception as e:
        logging.error(f"Error creating dashboard: {e}")
        return None

def select_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing JSON Files",
            initialdir=os.path.expanduser("~/Desktop")
        )
        return folder_path
    except Exception as e:
        logging.error(f"Error in folder selection: {e}")
        return None

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
            
        if len(data) < 2:
            logging.error("Insufficient data for analysis. Need at least 2 entries.")
            return

        preprocessed_data = preprocess_all_texts(data)
        if not preprocessed_data:
            logging.error("Failed to preprocess texts. Exiting.")
            return

        tokens, entities, dependencies, _ = extract_entities_and_tokens_parallel(preprocessed_data)
        if not tokens:
            logging.error("No tokens extracted. Exiting.")
            return

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
        
        app = create_dashboard(skills, job_matches, skill_evolution_df, topic_info, 
                             conversation_sentiments)
        if app:
            app.run_server(debug=True)
        else:
            logging.error("Failed to create dashboard")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

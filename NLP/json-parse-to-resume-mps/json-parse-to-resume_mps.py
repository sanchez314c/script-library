#!/usr/bin/env python3
"""
Enhanced AI Chat Resume Processor (Debug Version)
------------------------------------------------
Author: Jason Paul Michaels
Date: December 9, 2024
Version: 3.0.2

This version adds more logging and relaxed criteria to help debug why
the output might be empty.
"""

import logging
import os
import json
import spacy
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re

logging.basicConfig(
    level=logging.DEBUG,  # Increased logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self):
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) acceleration available")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("CUDA acceleration available")
            return torch.device("cuda")
        logger.info("Using CPU for processing")
        return torch.device("cpu")

class ModelManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.nlp = self._load_spacy()
        self.sentence_transformer = self._load_sentence_transformer()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
    
    def _load_spacy(self):
        try:
            return spacy.load("en_core_web_trf")
        except OSError:
            logger.info("Downloading SpaCy model...")
            os.system("python -m spacy download en_core_web_trf")
            return spacy.load("en_core_web_trf")
    
    def _load_sentence_transformer(self):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.device.type in ["mps", "cuda"]:
            model = model.to(self.device)
        return model
    
    def _load_sentiment_analyzer(self):
        device_id = -1 if self.device.type == "cpu" else 0
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device_id
        )

class ChatDataProcessor:
    def __init__(self, device_manager: DeviceManager, model_manager: ModelManager):
        self.device = device_manager.device
        self.models = model_manager
        self.skills_pattern = re.compile(
            r'\b(python|javascript|typescript|react|vue|angular|node\.js|sql|mongodb|aws|azure|docker|kubernetes|ci/cd|git|agile|scrum|machine learning|deep learning|nlp|computer vision|data science|api|rest|graphql|testing|devops|security)\b',
            re.IGNORECASE
        )
    
    def process_chat_data(self, file_path: str) -> Tuple[Dict, List, List]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            if not isinstance(chat_data, list):
                raise ValueError("JSON file does not contain a list of entries.")
            
            logger.info(f"Processing {len(chat_data)} chat entries")
            
            with ThreadPoolExecutor() as executor:
                skills_future = executor.submit(self._extract_skills, chat_data)
                achievements_future = executor.submit(self._extract_achievements, chat_data)
                timeline_future = executor.submit(self._create_timeline, chat_data)
                
                skills = skills_future.result()
                achievements = achievements_future.result()
                timeline = timeline_future.result()
            
            logger.debug(f"Final Skills Extracted: {skills}")
            logger.debug(f"Final Achievements Extracted: {achievements}")
            logger.debug(f"Final Timeline Extracted: {timeline}")
            
            return skills, achievements, timeline
            
        except Exception as e:
            logger.error(f"Error processing chat data: {e}")
            raise
    
    def _extract_skills(self, chat_data: List[Dict]) -> Dict[str, int]:
        skills_counter = Counter()
        
        for entry in chat_data:
            content = entry.get('content', '')
            if not content:
                continue
            found_skills = self.skills_pattern.findall(content.lower())
            skills_counter.update(found_skills)
            
            doc = self.models.nlp(content)
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'ORG', 'TECHNOLOGY']:
                    skills_counter[ent.text.lower()] += 1
        
        logger.debug(f"Skills found: {skills_counter}")
        return dict(skills_counter)
    
    def _extract_achievements(self, chat_data: List[Dict]) -> List[Dict]:
        achievements = []
        
        # Relaxing criteria: no length check, lower sentiment threshold
        achievement_patterns = [
            r'implement(?:ed|ing)?',
            r'develop(?:ed|ing)?',
            r'creat(?:ed|ing)?',
            r'build(?:ed|ing)?',
            r'design(?:ed|ing)?',
            r'optimi(?:zed|zing)',
            r'improv(?:ed|ing)'
        ]
        
        pattern = re.compile('|'.join(achievement_patterns), re.IGNORECASE)
        
        for entry in chat_data:
            content = entry.get('content', '')
            if not content:
                continue
            
            if pattern.search(content):
                # Lowering sentiment threshold to 0.5 for debugging
                sentiment = self.models.sentiment_analyzer(content)[0]
                if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.5:
                    date_str = entry.get('timestamp', '')
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                        date_val = date_obj.isoformat()
                    except:
                        date_val = date_str
                    achievements.append({
                        'description': content,
                        'date': date_val,
                        'sentiment_score': sentiment['score']
                    })
        
        logger.debug(f"Achievements found: {achievements}")
        return sorted(achievements, key=lambda x: x['sentiment_score'], reverse=True)[:10]
    
    def _create_timeline(self, chat_data: List[Dict]) -> List[Dict]:
        timeline = []
        
        # Relaxing event criteria: no content length check
        for entry in chat_data:
            content = entry.get('content', '')
            timestamp = entry.get('timestamp', '')
            if not content or not timestamp:
                continue
            
            # Remove complexity check and just categorize if there's content
            try:
                date_obj = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
                date_val = date_obj.isoformat()
            except:
                date_val = timestamp
            
            timeline.append({
                'date': date_val,
                'event': content[:200] + '...' if len(content) > 200 else content,
                'type': self._categorize_event(content)
            })
        
        logger.debug(f"Timeline events found: {timeline}")
        return sorted(timeline, key=lambda x: x['date'])
    
    def _categorize_event(self, content: str) -> str:
        categories = {
            'development': ['implement', 'develop', 'code', 'program'],
            'architecture': ['design', 'architect', 'structure'],
            'deployment': ['deploy', 'release', 'launch'],
            'optimization': ['optimize', 'improve', 'enhance']
        }
        
        content_lower = content.lower()
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'general'

class ResumeGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
    
    def generate_resume(self, name: str, skills: Dict[str, int], 
                        achievements: List[Dict], timeline: List[Dict], 
                        output_path: str):
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        self._add_header(story, name)
        self._add_summary(story, skills)
        self._add_skills_section(story, skills)
        self._add_achievements_section(story, achievements)
        self._add_timeline_section(story, timeline)
        self._add_visualizations(story, skills)
        
        doc.build(story)
        logger.info(f"Resume generated: {output_path}")
    
    def _add_header(self, story: List, name: str):
        story.append(Paragraph(name, self.styles['Title']))
        story.append(Paragraph('AI/ML Engineer & Technical Architect', 
                               self.styles['SubHeader']))
        story.append(Spacer(1, 20))
    
    def _add_summary(self, story: List, skills: Dict[str, int]):
        if skills:
            top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5]
            skill_text = ', '.join(skill for skill, _ in top_skills)
        else:
            skill_text = 'various technologies'
        
        summary = (
            f"Experienced AI/ML Engineer with expertise in {skill_text}. "
            "Demonstrated success in developing and implementing advanced AI "
            "solutions with a focus on natural language processing and machine "
            "learning systems. Proven track record of delivering innovative "
            "solutions and driving technological advancement."
        )
        
        story.append(Paragraph('Professional Summary', self.styles['SectionHeader']))
        story.append(Paragraph(summary, self.styles['Normal']))
        story.append(Spacer(1, 20))
    
    def _add_skills_section(self, story: List, skills: Dict[str, int]):
        story.append(Paragraph('Technical Skills', self.styles['SectionHeader']))
        
        if not skills:
            story.append(Paragraph("No technical skills identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        categories = {
            'AI/ML': [],
            'Programming': [],
            'Cloud & DevOps': [],
            'Tools & Frameworks': []
        }
        
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills:
            skill_entry = f"{skill.title()} ({count})"
            skill_lower = skill.lower()
            if any(term in skill_lower for term in ['ai', 'ml', 'learning', 'neural', 'nlp']):
                categories['AI/ML'].append(skill_entry)
            elif any(term in skill_lower for term in ['python', 'java', 'javascript', 'code']):
                categories['Programming'].append(skill_entry)
            elif any(term in skill_lower for term in ['aws', 'azure', 'docker','kubernetes']):
                categories['Cloud & DevOps'].append(skill_entry)
            else:
                categories['Tools & Frameworks'].append(skill_entry)
        
        max_rows = max((len(c) for c in categories.values()), default=1)
        table_data = [list(categories.keys())]
        
        for i in range(max_rows):
            row = []
            for category in categories:
                cell = categories[category][i] if i < len(categories[category]) else ''
                row.append(cell)
            table_data.append(row)
        
        from reportlab.lib import colors
        table = Table(table_data, colWidths=[150] * len(categories))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_achievements_section(self, story: List, achievements: List[Dict]):
        story.append(Paragraph('Key Achievements', self.styles['SectionHeader']))
        
        if not achievements:
            story.append(Paragraph("No notable achievements found.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        for achievement in achievements:
            description = achievement['description']
            if len(description) > 200:
                description = description[:197] + '...'
            
            date_str = achievement.get('date', '')
            date_formatted = date_str
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str)
                    date_formatted = date_obj.strftime('%b %Y')
                except:
                    pass
            
            entry = f"• {date_formatted}: {description}"
            story.append(Paragraph(entry, self.styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    def _add_timeline_section(self, story: List, timeline: List[Dict]):
        story.append(Paragraph('Professional Timeline', self.styles['SectionHeader']))
        
        if not timeline:
            story.append(Paragraph("No timeline events identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        table_data = [['Date', 'Event', 'Type']]
        
        events_sorted = sorted(timeline, key=lambda x: x['date'])
        
        for event in events_sorted[:10]:
            date_str = event['date']
            date_formatted = date_str
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str)
                    date_formatted = date_obj.strftime('%b %Y')
                except:
                    pass
            
            event_text = event['event']
            if len(event_text) > 100:
                event_text = event_text[:97] + '...'
            
            table_data.append([
                date_formatted,
                event_text,
                event['type'].title()
            ])
        
        table = Table(table_data, colWidths=[1*inch, 4*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_visualizations(self, story: List, skills: Dict[str, int]):
        story.append(Paragraph('Skill Distribution', self.styles['SectionHeader']))
        
        if not skills:
            story.append(Paragraph("No skills available for visualization.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        categories = {
            'AI/ML': [],
            'Programming': [],
            'Cloud & DevOps': [],
            'Tools & Frameworks': []
        }
        
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills:
            skill_lower = skill.lower()
            if any(term in skill_lower for term in ['ai', 'ml', 'learning', 'neural', 'nlp']):
                categories['AI/ML'].append((skill, count))
            elif any(term in skill_lower for term in ['python', 'java', 'javascript', 'code']):
                categories['Programming'].append((skill, count))
            elif any(term in skill_lower for term in ['aws', 'azure', 'docker','kubernetes']):
                categories['Cloud & DevOps'].append((skill, count))
            else:
                categories['Tools & Frameworks'].append((skill, count))
        
        top_skills = []
        for category, skill_list in categories.items():
            top_skills.extend([(category, s, c) for s, c in skill_list])
        
        top_skills = sorted(top_skills, key=lambda x: x[2], reverse=True)[:20]
        
        if not top_skills:
            story.append(Paragraph("No skills matched categorization or available for visualization.", self.styles['Normal']))
            story.append(Spacer(1,20))
            return
        
        cat_labels = []
        skill_names = []
        counts = []
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        category_keys = list(categories.keys())
        
        for category, s, c in top_skills:
            cat_labels.append(category)
            skill_names.append(s)
            counts.append(c)
        
        try:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(skill_names))
            bars = plt.bar(x, counts, color=[colors_list[category_keys.index(cat)] for cat in cat_labels])
            plt.xticks(x, skill_names, rotation=45, ha='right')
            plt.ylabel('Frequency')
            plt.title('Skills Distribution by Category')
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor=colors_list[i], label=category_keys[i]) 
                for i in range(len(category_keys))
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.tight_layout()
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            story.append(Image(img_buffer, width=500, height=300))
            story.append(Spacer(1, 20))
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            story.append(Paragraph("Skill distribution visualization unavailable", 
                                   self.styles['Normal']))
            story.append(Spacer(1, 20))

def main():
    try:
        device_manager = DeviceManager()
        model_manager = ModelManager(device_manager.device)
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Chat JSON File",
            filetypes=[("JSON files", "*.json")]
        )
        
        if not file_path:
            logger.error("No file selected")
            return
        
        processor = ChatDataProcessor(device_manager, model_manager)
        skills, achievements, timeline = processor.process_chat_data(file_path)
        
        name = input("Enter your name for the resume: ").strip()
        if not name:
            name = "User"
        output_path = f"{name.replace(' ', '_')}_AI_Resume.pdf"
        
        resume_gen = ResumeGenerator()
        resume_gen.generate_resume(name, skills, achievements, timeline, output_path)
        
        logger.info(f"Resume generated successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

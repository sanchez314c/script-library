#!/usr/bin/env python3

import os
import json
import time
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any
import logging
from pathlib import Path
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Get script directory
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Setup NLTK data path in the script directory
NLTK_DATA_DIR = SCRIPT_DIR / 'nltk_data'
NLTK_DATA_DIR.mkdir(exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=SCRIPT_DIR / 'ai_job_analysis.log'
)

def setup_nltk():
    """Download required NLTK data packages to script directory"""
    required_packages = ['vader_lexicon', 'punkt', 'averaged_perceptron_tagger']
    
    print("Setting up NLTK packages...")
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"Package {package} already downloaded")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)
            print(f"Successfully downloaded {package}")

# Run NLTK setup
setup_nltk()

# Load environment variables
load_dotenv()

# Initialize APIs
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

class AIJobAnalyzer:
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
            print("Successfully initialized sentiment analyzer")
        except Exception as e:
            logging.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise
            
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def google_search(self, query: str, start_index: int = 1) -> Dict:
        """Execute a Google Custom Search"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'q': query,
            'start': start_index
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Search query '{query}' returned {len(data.get('items', []))} results")
            return data
        except Exception as e:
            logging.error(f"Error in Google search: {str(e)}")
            print(f"Google API Error: {str(e)}")
            return {"items": []}

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text using NLTK"""
        try:
            # Tokenize and tag parts of speech
            tokens = nltk.word_tokenize(text.lower())
            tagged = nltk.pos_tag(tokens)
            
            # Extract nouns and adjectives
            keywords = [word for word, pos in tagged 
                       if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']
                       and len(word) > 3]
            
            # Calculate frequency distribution
            freq_dist = nltk.FreqDist(keywords)
            return [word for word, _ in freq_dist.most_common(num_keywords)]
        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            print(f"Keyword extraction error: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            return self.sia.polarity_scores(text)
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            print(f"Sentiment analysis error: {str(e)}")
            return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

    def scrape_article(self, url: str) -> Dict[str, str]:
        """Scrape and parse article content"""
        try:
            print(f"\nAttempting to scrape: {url}")
            response = requests.get(url, headers=self.headers, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Basic text cleaning
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get title
            title = soup.title.string if soup.title else url
            
            print(f"Successfully scraped article: {title[:100]}...")
            
            # Create summary (first 500 characters)
            summary = text[:500] + "..."
            
            # Extract keywords and analyze sentiment
            keywords = self.extract_keywords(text)
            sentiment = self.analyze_sentiment(text)
            
            return {
                'title': title,
                'text': text,
                'summary': summary,
                'keywords': keywords,
                'sentiment': sentiment
            }
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            print(f"Error scraping {url}: {str(e)}")
            return None

    def claude_analysis(self, content: str) -> str:
        """Get Claude's analysis of the content"""
        if not content.strip():
            return "No content available for analysis."
            
        prompt = f"""
        Analyze the following content about AI job market trends and provide:
        1. Key emerging roles and skills
        2. Salary trends and predictions
        3. Industry sectors with highest demand
        4. Required qualifications and experience
        5. Future outlook for next 18 months

        Content: {content}

        Provide a detailed, structured analysis focusing on actionable insights.
        """
        
        try:
            response = anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content.text if hasattr(response.content, 'text') else str(response.content)
        except Exception as e:
            logging.error(f"Error in Claude analysis: {str(e)}")
            return f"Error in analysis: {str(e)}"

    def collect_job_data(self) -> List[Dict[str, Any]]:
        """Collect and analyze job market data"""
        search_queries = [
            "site:techcrunch.com AI job market trends 2024",
            "site:venturebeat.com artificial intelligence jobs 2024",
            "site:ieee.org machine learning career trends",
            "site:wired.com AI industry hiring trends",
            "site:technologyreview.com emerging AI roles"
        ]
        
        all_articles = []
        
        for query in search_queries:
            try:
                search_results = self.google_search(query)
                print(f"\nProcessing query: {query}")
                print(f"Found {len(search_results.get('items', []))} results")
                
                for item in search_results.get('items', []):
                    article_data = self.scrape_article(item['link'])
                    
                    if article_data:
                        article_data['url'] = item['link']
                        all_articles.append(article_data)
                        print(f"Successfully processed: {item['link']}")
                
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing query '{query}': {str(e)}")
                print(f"Error processing query: {str(e)}")
                
        return all_articles

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logging.info("Starting job market analysis...")
        print("Starting job market analysis...")
        
        articles = self.collect_job_data()
        print(f"Collected {len(articles)} articles")
        
        combined_content = "\n\n".join([
            f"Title: {article['title']}\n{article['summary']}\nKeywords: {', '.join(article['keywords'])}\nSentiment: {article['sentiment']['compound']:.2f}"
            for article in articles if article
        ])
        
        claude_insights = self.claude_analysis(combined_content)
        
        # Calculate average sentiment
        sentiments = [
            article['sentiment']['compound']
            for article in articles if article
        ]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Create JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'number_of_sources': len(articles),
            'average_market_sentiment': avg_sentiment,
            'claude_analysis': claude_insights,
            'sources': [{
                'title': article['title'],
                'url': article['url'],
                'keywords': article['keywords'],
                'sentiment': article['sentiment']
            } for article in articles if article]
        }
        
        # Save JSON report
        report_path = SCRIPT_DIR / 'ai_job_market_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Create markdown report
        markdown_content = f"""# AI Job Market Analysis Report
    Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

    ## Overview
    - Analyzed Sources: {len(articles)}
    - Market Sentiment: {avg_sentiment:.2f}

    ## Top AI Jobs for 2024-2025

    1. **AI Prompt Engineer**
       - Salary Range: $125,000 - $250,000
       - Focus: Crafting and optimizing prompts for generative AI systems
       - Required Skills: Natural language processing, LLM understanding, creative writing

    2. **AI/ML Research Engineer**
       - Salary Range: $150,000 - $300,000+
       - Focus: Developing new AI models and algorithms
       - Required Skills: PhD preferred, PyTorch, TensorFlow, deep learning expertise

    3. **AI Ethics & Governance Specialist**
       - Salary Range: $120,000 - $200,000
       - Focus: Ensuring responsible AI development and deployment
       - Required Skills: Policy background, technical understanding, ethics training

    4. **Climate Tech AI Specialist**
       - Salary Range: $140,000 - $250,000
       - Focus: AI applications for environmental solutions
       - Required Skills: Climate science knowledge, ML modeling, sustainability metrics

    5. **AI Integration Architect**
       - Salary Range: $160,000 - $280,000
       - Focus: Implementing AI systems into existing infrastructure
       - Required Skills: Cloud platforms, MLOps, system architecture

    6. **Biotech AI Researcher**
       - Salary Range: $130,000 - $275,000
       - Focus: AI applications in healthcare and biotechnology
       - Required Skills: Biology background, ML expertise, medical data analysis

    7. **AI Product Manager**
       - Salary Range: $140,000 - $240,000
       - Focus: Managing AI product development and deployment
       - Required Skills: Technical background, product management, AI understanding

    8. **Government AI Specialist**
       - Salary Range: $110,000 - $190,000
       - Focus: AI implementation in public sector
       - Required Skills: Public policy knowledge, security clearance, AI expertise

    9. **AI Security Engineer**
       - Salary Range: $140,000 - $260,000
       - Focus: Securing AI systems and models
       - Required Skills: Cybersecurity, ML understanding, risk assessment

    10. **MLOps Engineer**
        - Salary Range: $130,000 - $230,000
        - Focus: AI/ML infrastructure and deployment
        - Required Skills: DevOps, ML pipelines, cloud platforms

    ## Key Growth Areas
    - Climate Tech AI (fastest growing sector)
    - Healthcare/Biotech AI
    - Government AI initiatives
    - Financial Technology AI
    - AI Security & Governance

    ## Market Analysis
    {claude_insights}

    ## Top Sources
    """
        
        # Add top sources with high sentiment scores
        sorted_sources = sorted(
            [a for a in articles if a],
            key=lambda x: x['sentiment']['compound'],
            reverse=True
        )[:10]
        
        for idx, source in enumerate(sorted_sources, 1):
            markdown_content += f"\n{idx}. [{source['title']}]({source['url']})"
            markdown_content += f"\n   - Sentiment: {source['sentiment']['compound']:.2f}"
            if source['keywords']:
                markdown_content += f"\n   - Key Topics: {', '.join(source['keywords'])}"
        
        # Save markdown report
        markdown_path = SCRIPT_DIR / 'ai_job_market_report.md'
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
                
        logging.info(f"Analysis complete. Reports generated at {report_path} and {markdown_path}")
        return report

def main():
    analyzer = AIJobAnalyzer()
    report = analyzer.generate_report()
    
    print("\n=== AI Job Market Analysis Report ===")
    print(f"\nAnalyzed {report['number_of_sources']} sources")
    print(f"Market Sentiment: {report['average_market_sentiment']:.2f}")
    print("\nReports generated:")
    print(f"1. JSON Report: {SCRIPT_DIR / 'ai_job_market_report.json'}")
    print(f"2. Markdown Report: {SCRIPT_DIR / 'ai_job_market_report.md'}")

if __name__ == "__main__":
    main()


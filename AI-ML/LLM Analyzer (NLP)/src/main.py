"""
Main execution script for Cortana Story Analysis
"""

from data_processor import DataProcessor
from ml_analyzer import MLAnalyzer
from story_generator import StoryGenerator
import argparse
import logging
import json

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze Cortana Story development")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to input data directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    args = parser.parse_args()
    
    # Initialize components
    processor = DataProcessor(args.config)
    analyzer = MLAnalyzer(args.config)
    generator = StoryGenerator(args.config)
    
    # Execute pipeline
    dataset = processor.create_dataset()
    analysis_results = analyzer.analyze_all(dataset)
    story = generator.generate_narrative(analysis_results)
    
if __name__ == "__main__":
    main()

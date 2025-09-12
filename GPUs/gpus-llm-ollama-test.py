#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: gpus-llm-ollama-test.py                                           
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive Ollama LLM testing and benchmarking tool with      
#              GPU acceleration validation, performance profiling, model        
#              compatibility testing, and response quality assessment.          
#
# Usage: python gpus-llm-ollama-test.py [--endpoint URL] [--model NAME] [--bench]
#
# Dependencies: requests, torch, psutil, ollama-python                         
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional Ollama integration testing with GPU acceleration         
#        validation, comprehensive benchmarking, and development optimization.  
#                                                                                
####################################################################################

"""
Comprehensive Ollama LLM Testing Tool
====================================

Advanced testing and benchmarking tool for Ollama LLM deployments with GPU
acceleration validation, performance profiling, model compatibility testing,
and comprehensive response quality assessment for development workflows.
"""

import os
import sys
import logging
import subprocess
import argparse
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import statistics
from datetime import datetime
import threading

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "requests>=2.28.0",
        "torch>=1.12.0",
        "psutil>=5.8.0",
        "ollama>=0.1.0"
    ]
    
    missing = []
    for pkg in required_packages:
        name = pkg.split('>=')[0].replace('-', '_')
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

check_and_install_dependencies()

import tkinter as tk
from tkinter import messagebox, ttk
import torch
import psutil

try:
    import ollama
    OLLAMA_PYTHON_AVAILABLE = True
except ImportError:
    OLLAMA_PYTHON_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class OllamaLLMTester:
    def __init__(self):
        self.setup_logging()
        self.endpoint = "http://localhost:11434"
        self.test_results = {}
        self.performance_metrics = {}
        self.model_list = []
        
        # Test prompts for different capabilities
        self.test_prompts = {
            'basic': "Hello! How are you today?",
            'reasoning': "If I have 3 apples and I give away 1, how many do I have left? Explain your reasoning.",
            'creative': "Write a short haiku about artificial intelligence.",
            'code': "Write a simple Python function to calculate the factorial of a number.",
            'complex': "Explain the concept of quantum entanglement in simple terms that a high school student could understand."
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-llm-ollama-test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def check_ollama_service(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Ollama service is running")
                return True
            else:
                self.logger.error(f"❌ Ollama service returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Cannot connect to Ollama service")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error checking Ollama service: {e}")
            return False
            
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self.model_list = models
                self.logger.info(f"📋 Found {len(models)} available models")
                return models
            else:
                self.logger.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            return []
            
    def test_model_response(self, model_name: str, prompt: str, timeout: int = 30):
        """Test model response with timing"""
        try:
            start_time = time.time()
            
            # Use Ollama Python client if available
            if OLLAMA_PYTHON_AVAILABLE:
                try:
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        stream=False
                    )
                    end_time = time.time()
                    
                    result = {
                        'success': True,
                        'response': response.get('response', ''),
                        'response_time': end_time - start_time,
                        'tokens': len(response.get('response', '').split()),
                        'model': response.get('model', model_name)
                    }
                    return result
                except Exception as e:
                    self.logger.warning(f"Ollama Python client failed, trying API: {e}")
            
            # Fallback to direct API
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    'success': True,
                    'response': data.get('response', ''),
                    'response_time': end_time - start_time,
                    'tokens': len(data.get('response', '').split()),
                    'model': data.get('model', model_name),
                    'eval_count': data.get('eval_count', 0),
                    'eval_duration': data.get('eval_duration', 0)
                }
                
                # Calculate tokens per second if available
                if result['eval_count'] > 0 and result['eval_duration'] > 0:
                    result['tokens_per_second'] = result['eval_count'] / (result['eval_duration'] / 1e9)
                    
                return result
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': end_time - start_time
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f"Request timeout after {timeout} seconds",
                'response_time': timeout
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
            
    def benchmark_model_performance(self, model_name: str, iterations: int = 5):
        """Benchmark model performance with multiple iterations"""
        self.logger.info(f"🚀 Benchmarking {model_name} performance...")
        
        benchmark_results = {
            'model': model_name,
            'iterations': iterations,
            'tests': {},
            'summary': {}
        }
        
        for test_name, prompt in self.test_prompts.items():
            self.logger.info(f"   Testing {test_name} capability...")
            test_results = []
            
            for i in range(iterations):
                result = self.test_model_response(model_name, prompt)
                test_results.append(result)
                
                if result['success']:
                    self.logger.info(f"     Iteration {i+1}: {result['response_time']:.2f}s")
                else:
                    self.logger.error(f"     Iteration {i+1}: Failed - {result.get('error', 'Unknown error')}")
                    
            # Calculate statistics
            successful_tests = [r for r in test_results if r['success']]
            
            if successful_tests:
                response_times = [r['response_time'] for r in successful_tests]
                tokens_counts = [r['tokens'] for r in successful_tests]
                
                test_summary = {
                    'success_rate': len(successful_tests) / len(test_results),
                    'avg_response_time': statistics.mean(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'std_response_time': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    'avg_tokens': statistics.mean(tokens_counts),
                    'sample_response': successful_tests[0]['response'][:200] + "..." if len(successful_tests[0]['response']) > 200 else successful_tests[0]['response']
                }
                
                # Add tokens per second if available
                tokens_per_sec = [r.get('tokens_per_second', 0) for r in successful_tests if r.get('tokens_per_second', 0) > 0]
                if tokens_per_sec:
                    test_summary['avg_tokens_per_second'] = statistics.mean(tokens_per_sec)
                    
            else:
                test_summary = {
                    'success_rate': 0,
                    'error': 'All iterations failed'
                }
                
            benchmark_results['tests'][test_name] = {
                'results': test_results,
                'summary': test_summary
            }
            
        # Overall summary
        all_successful = []
        for test_data in benchmark_results['tests'].values():
            if 'summary' in test_data and 'success_rate' in test_data['summary']:
                all_successful.extend([r for r in test_data['results'] if r['success']])
                
        if all_successful:
            all_response_times = [r['response_time'] for r in all_successful]
            benchmark_results['summary'] = {
                'total_tests': len(all_successful),
                'overall_avg_response_time': statistics.mean(all_response_times),
                'overall_success_rate': len(all_successful) / (len(self.test_prompts) * iterations)
            }
            
        return benchmark_results
        
    def check_gpu_utilization(self):
        """Monitor GPU utilization during testing"""
        gpu_info = {'available': False}
        
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['devices'] = []
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': device_props.name,
                        'total_memory': device_props.total_memory,
                        'memory_allocated': torch.cuda.memory_allocated(i),
                        'memory_reserved': torch.cuda.memory_reserved(i),
                        'utilization_percent': (torch.cuda.memory_allocated(i) / device_props.total_memory) * 100
                    }
                    gpu_info['devices'].append(device_info)
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['available'] = True
                gpu_info['device_type'] = 'Apple Silicon MPS'
                gpu_info['device_count'] = 1
                
        except Exception as e:
            self.logger.error(f"Error checking GPU utilization: {e}")
            gpu_info['error'] = str(e)
            
        return gpu_info
        
    def test_model_compatibility(self, model_name: str):
        """Test model compatibility and capabilities"""
        compatibility_tests = {
            'basic_response': False,
            'handles_long_prompt': False,
            'json_output': False,
            'code_generation': False,
            'multilingual': False
        }
        
        # Basic response test
        result = self.test_model_response(model_name, "Hello", timeout=10)
        compatibility_tests['basic_response'] = result['success']
        
        if result['success']:
            # Long prompt test
            long_prompt = "Please explain the following concept in detail: " + "artificial intelligence " * 50
            result = self.test_model_response(model_name, long_prompt, timeout=30)
            compatibility_tests['handles_long_prompt'] = result['success']
            
            # JSON output test
            json_prompt = "Respond with a JSON object containing the keys 'name' and 'age' for a fictional character."
            result = self.test_model_response(model_name, json_prompt, timeout=15)
            compatibility_tests['json_output'] = result['success'] and '{' in result.get('response', '')
            
            # Code generation test
            code_prompt = "Write a Python function to sort a list of numbers."
            result = self.test_model_response(model_name, code_prompt, timeout=20)
            compatibility_tests['code_generation'] = result['success'] and 'def ' in result.get('response', '')
            
            # Multilingual test (Spanish)
            spanish_prompt = "Hola, ¿cómo estás? Por favor responde en español."
            result = self.test_model_response(model_name, spanish_prompt, timeout=15)
            compatibility_tests['multilingual'] = result['success']
            
        return compatibility_tests
        
    def create_progress_window(self, total_tests):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Ollama LLM Testing")
        self.progress_root.geometry("500x150")
        self.progress_root.resizable(False, False)
        
        # Center window
        x = (self.progress_root.winfo_screenwidth() // 2) - 250
        y = (self.progress_root.winfo_screenheight() // 2) - 75
        self.progress_root.geometry(f"+{x}+{y}")
        
        if IS_MACOS:
            self.progress_root.attributes("-topmost", True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_root,
            variable=self.progress_var,
            maximum=total_tests,
            length=450
        )
        self.progress_bar.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing Ollama testing...")
        self.status_label = tk.Label(
            self.progress_root,
            textvariable=self.status_var,
            font=("Helvetica", 11)
        )
        self.status_label.pack(pady=(0, 20))
        
        self.progress_root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.progress_root.update()
        
    def update_progress(self, current, total, status=""):
        """Update progress bar and status"""
        if hasattr(self, 'progress_var'):
            self.progress_var.set(current)
            
            percentage = int((current / total) * 100)
            if status:
                self.status_var.set(f"{status} ({percentage}%)")
            else:
                self.status_var.set(f"Testing {current}/{total} ({percentage}%)")
                
            self.progress_root.update()
            
    def generate_report(self, output_path=None):
        """Generate comprehensive test report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = Path.home() / "Desktop" / f"ollama_test_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        report = {
            'timestamp': timestamp,
            'endpoint': self.endpoint,
            'available_models': self.model_list,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'gpu_info': self.check_gpu_utilization(),
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__ if 'torch' in sys.modules else 'Not available',
                'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📄 Report saved to: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return None
            
    def display_summary(self):
        """Display test results summary"""
        print("\n" + "="*70)
        print("              OLLAMA LLM TESTING SUMMARY")
        print("="*70)
        
        print(f"\n🔗 Endpoint: {self.endpoint}")
        print(f"📋 Available Models: {len(self.model_list)}")
        
        if self.model_list:
            for model in self.model_list:
                print(f"   • {model}")
                
        # Test Results Summary
        if self.test_results:
            print(f"\n🧪 TEST RESULTS:")
            for model_name, results in self.test_results.items():
                print(f"\n   📱 Model: {model_name}")
                
                if 'compatibility' in results:
                    compatibility = results['compatibility']
                    passed = sum(1 for v in compatibility.values() if v)
                    total = len(compatibility)
                    print(f"      Compatibility: {passed}/{total} tests passed")
                    
                if 'benchmark' in results and 'summary' in results['benchmark']:
                    summary = results['benchmark']['summary']
                    print(f"      Success Rate: {summary.get('overall_success_rate', 0)*100:.1f}%")
                    print(f"      Avg Response Time: {summary.get('overall_avg_response_time', 0):.2f}s")
                    
        # GPU Information
        gpu_info = self.check_gpu_utilization()
        print(f"\n🚀 GPU INFORMATION:")
        if gpu_info.get('available'):
            print(f"   Status: ✅ Available")
            if 'device_type' in gpu_info:
                print(f"   Type: {gpu_info['device_type']}")
            elif 'devices' in gpu_info:
                for device in gpu_info['devices']:
                    print(f"   Device {device['id']}: {device['name']}")
                    print(f"      Memory Usage: {device['utilization_percent']:.1f}%")
        else:
            print(f"   Status: ❌ Not Available")
            
        print("\n" + "="*70)
        
    def show_gui_summary(self):
        """Show GUI summary of test results"""
        if not IS_MACOS:
            return
            
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Create summary message
            models_count = len(self.model_list)
            tests_run = len(self.test_results)
            
            summary = f"""Ollama LLM Testing Summary

Endpoint: {self.endpoint}
Available Models: {models_count}
Models Tested: {tests_run}

"""
            
            if self.test_results:
                summary += "Test Results:\n"
                for model_name, results in self.test_results.items():
                    if 'benchmark' in results and 'summary' in results['benchmark']:
                        success_rate = results['benchmark']['summary'].get('overall_success_rate', 0) * 100
                        avg_time = results['benchmark']['summary'].get('overall_avg_response_time', 0)
                        summary += f"• {model_name}: {success_rate:.1f}% success, {avg_time:.2f}s avg\n"
                        
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("Ollama Testing Results", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive Ollama LLM Testing Tool")
        parser.add_argument('--endpoint', '-e', default="http://localhost:11434", help='Ollama endpoint URL')
        parser.add_argument('--model', '-m', help='Specific model to test')
        parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmarks')
        parser.add_argument('--iterations', '-i', type=int, default=3, help='Number of benchmark iterations')
        parser.add_argument('--prompt', '-p', help='Custom test prompt')
        parser.add_argument('--export', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Ollama LLM Testing Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.endpoint = args.endpoint
        self.logger.info(f"🚀 Starting Ollama LLM testing on {self.endpoint}...")
        
        try:
            # Check Ollama service
            if not self.check_ollama_service():
                print("❌ Ollama service is not running or not accessible")
                print(f"   Please ensure Ollama is running on {self.endpoint}")
                return
                
            # Get available models
            self.get_available_models()
            
            if not self.model_list:
                print("❌ No models found. Please install a model first:")
                print("   ollama pull llama2")
                return
                
            # Determine models to test
            models_to_test = [args.model] if args.model else self.model_list[:3]  # Test first 3 models
            
            total_tests = len(models_to_test) * 2  # Compatibility + benchmark
            current_test = 0
            
            # Create progress window
            self.create_progress_window(total_tests)
            
            # Test each model
            for model_name in models_to_test:
                self.logger.info(f"\n🧪 Testing model: {model_name}")
                
                # Update progress
                self.update_progress(current_test, total_tests, f"Testing {model_name} compatibility")
                current_test += 1
                
                # Test compatibility
                compatibility_results = self.test_model_compatibility(model_name)
                
                model_results = {
                    'compatibility': compatibility_results
                }
                
                # Run benchmarks if requested
                if args.benchmark:
                    self.update_progress(current_test, total_tests, f"Benchmarking {model_name}")
                    benchmark_results = self.benchmark_model_performance(model_name, args.iterations)
                    model_results['benchmark'] = benchmark_results
                    
                # Custom prompt test if provided
                if args.prompt:
                    custom_result = self.test_model_response(model_name, args.prompt)
                    model_results['custom_prompt'] = custom_result
                    
                self.test_results[model_name] = model_results
                current_test += 1
                
            # Close progress window
            if hasattr(self, 'progress_root'):
                self.progress_root.destroy()
                
            # Display results
            self.display_summary()
            
            # Generate report
            report_path = self.generate_report(args.export)
            
            # Show GUI summary if requested
            if args.gui:
                self.show_gui_summary()
                
            # Open files if on macOS
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)
                if report_path:
                    subprocess.run(['open', str(report_path)], check=False)
                    
        except KeyboardInterrupt:
            self.logger.info("Testing interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

def main():
    tester = OllamaLLMTester()
    tester.run()

if __name__ == "__main__":
    main()
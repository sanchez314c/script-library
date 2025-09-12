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
# Script Name: gpus-llm-openwebui-test.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive OpenWebUI testing and validation tool with API     
#              endpoint testing, model availability verification, streaming     
#              capability assessment, and performance benchmarking.             
#
# Usage: python gpus-llm-openwebui-test.py [--endpoint URL] [--api-key KEY]     
#
# Dependencies: requests, websockets, torch, aiohttp                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional OpenWebUI integration testing with comprehensive API      
#        validation, streaming assessment, and development optimization.        
#                                                                                
####################################################################################

"""
Comprehensive OpenWebUI Testing Tool
===================================

Advanced testing and validation tool for OpenWebUI deployments with API endpoint
testing, model availability verification, streaming capability assessment, and
comprehensive performance benchmarking for development workflows.
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
import asyncio
import threading

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "requests>=2.28.0",
        "websockets>=10.0",
        "torch>=1.12.0",
        "aiohttp>=3.8.0"
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
import websockets
import aiohttp

IS_MACOS = sys.platform == "darwin"

class OpenWebUITester:
    def __init__(self):
        self.setup_logging()
        self.endpoint = "http://localhost:3000"
        self.api_key = None
        self.test_results = {}
        self.api_endpoints = {}
        self.model_list = []
        self.session = None
        
        # Test configurations
        self.test_prompts = {
            'simple': "Hello, how are you?",
            'reasoning': "What is 2+2? Explain your calculation step by step.",
            'creative': "Write a short poem about technology.",
            'technical': "Explain the concept of machine learning in one paragraph.",
            'long_form': "Describe the history and development of artificial intelligence in detail."
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-llm-openwebui-test.log"
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
        
    def setup_session(self):
        """Setup HTTP session with proper headers"""
        self.session = requests.Session()
        
        # Set up headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'OpenWebUI-Tester/1.0.0'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        self.session.headers.update(headers)
        
    def check_service_availability(self):
        """Check if OpenWebUI service is accessible"""
        try:
            # Try to access the main page
            response = self.session.get(self.endpoint, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("✅ OpenWebUI service is accessible")
                return True
            else:
                self.logger.warning(f"⚠️ OpenWebUI returned status {response.status_code}")
                return True  # Service is running but might need authentication
                
        except requests.exceptions.ConnectionError:
            self.logger.error("❌ Cannot connect to OpenWebUI service")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error checking OpenWebUI service: {e}")
            return False
            
    def discover_api_endpoints(self):
        """Discover available API endpoints"""
        common_endpoints = {
            'api_info': '/api/v1/info',
            'models': '/api/v1/models',
            'chat_completions': '/api/v1/chat/completions',
            'completions': '/api/v1/completions',
            'health': '/health',
            'status': '/api/status',
            'config': '/api/config',
            'auth_signin': '/api/v1/auths/signin',
            'users': '/api/v1/users',
            'ollama_api': '/ollama/api/tags'
        }
        
        discovered = {}
        
        for endpoint_name, path in common_endpoints.items():
            try:
                url = f"{self.endpoint}{path}"
                response = self.session.get(url, timeout=5)
                
                discovered[endpoint_name] = {
                    'path': path,
                    'status_code': response.status_code,
                    'accessible': response.status_code in [200, 401, 403],  # 401/403 means endpoint exists
                    'response_size': len(response.content),
                    'content_type': response.headers.get('content-type', 'unknown')
                }
                
                if response.status_code == 200:
                    self.logger.info(f"✅ Found endpoint: {endpoint_name} ({path})")
                elif response.status_code in [401, 403]:
                    self.logger.info(f"🔒 Found protected endpoint: {endpoint_name} ({path})")
                    
            except Exception as e:
                discovered[endpoint_name] = {
                    'path': path,
                    'accessible': False,
                    'error': str(e)
                }
                
        self.api_endpoints = discovered
        accessible_count = sum(1 for ep in discovered.values() if ep.get('accessible', False))
        self.logger.info(f"📋 Discovered {accessible_count}/{len(common_endpoints)} API endpoints")
        
        return discovered
        
    def get_available_models(self):
        """Get list of available models"""
        models = []
        
        # Try OpenAI-compatible models endpoint
        try:
            response = self.session.get(f"{self.endpoint}/api/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    models.extend([model['id'] for model in data['data']])
                    self.logger.info(f"✅ Found {len(models)} models via OpenAI API")
        except Exception as e:
            self.logger.debug(f"OpenAI models endpoint failed: {e}")
            
        # Try Ollama models endpoint
        try:
            response = self.session.get(f"{self.endpoint}/ollama/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'models' in data:
                    ollama_models = [model['name'] for model in data['models']]
                    models.extend(ollama_models)
                    self.logger.info(f"✅ Found {len(ollama_models)} Ollama models")
        except Exception as e:
            self.logger.debug(f"Ollama models endpoint failed: {e}")
            
        # Try custom models endpoint
        try:
            response = self.session.get(f"{self.endpoint}/api/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    custom_models = [model.get('name', model.get('id', str(model))) for model in data]
                    models.extend(custom_models)
                    self.logger.info(f"✅ Found {len(custom_models)} custom models")
        except Exception as e:
            self.logger.debug(f"Custom models endpoint failed: {e}")
            
        # Remove duplicates and store
        self.model_list = list(set(models))
        return self.model_list
        
    def test_chat_completion(self, model: str, prompt: str, stream: bool = False):
        """Test chat completion endpoint"""
        try:
            start_time = time.time()
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": stream,
                "max_tokens": 500
            }
            
            response = self.session.post(
                f"{self.endpoint}/api/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    content = ""
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]
                                if data_str != '[DONE]':
                                    try:
                                        chunk_data = json.loads(data_str)
                                        delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                                        if 'content' in delta:
                                            content += delta['content']
                                    except json.JSONDecodeError:
                                        continue
                    
                    result = {
                        'success': True,
                        'response': content,
                        'response_time': end_time - start_time,
                        'tokens': len(content.split()),
                        'streaming': True
                    }
                else:
                    # Handle regular response
                    data = response.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    result = {
                        'success': True,
                        'response': content,
                        'response_time': end_time - start_time,
                        'tokens': len(content.split()),
                        'streaming': False,
                        'usage': data.get('usage', {}),
                        'model': data.get('model', model)
                    }
                    
                return result
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': end_time - start_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
            
    def test_websocket_connection(self):
        """Test WebSocket connection for real-time features"""
        async def test_ws():
            try:
                # Convert HTTP URL to WebSocket URL
                ws_url = self.endpoint.replace('http://', 'ws://').replace('https://', 'wss://')
                ws_url += '/ws'
                
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    # Send a test message
                    test_message = {"type": "ping", "data": "test"}
                    await websocket.send(json.dumps(test_message))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    
                    return {
                        'success': True,
                        'response': response,
                        'url': ws_url
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'url': ws_url if 'ws_url' in locals() else 'unknown'
                }
                
        try:
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test_ws())
            loop.close()
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"WebSocket test failed: {e}"
            }
            
    def benchmark_model_performance(self, model: str, iterations: int = 3):
        """Benchmark model performance with multiple test types"""
        self.logger.info(f"🚀 Benchmarking {model} performance...")
        
        benchmark_results = {
            'model': model,
            'iterations': iterations,
            'tests': {},
            'summary': {}
        }
        
        for test_name, prompt in self.test_prompts.items():
            self.logger.info(f"   Testing {test_name} capability...")
            test_results = []
            
            for i in range(iterations):
                # Test both streaming and non-streaming
                result_stream = self.test_chat_completion(model, prompt, stream=True)
                result_normal = self.test_chat_completion(model, prompt, stream=False)
                
                test_results.append({
                    'iteration': i + 1,
                    'streaming': result_stream,
                    'normal': result_normal
                })
                
                if result_normal['success']:
                    self.logger.info(f"     Iteration {i+1}: {result_normal['response_time']:.2f}s")
                else:
                    self.logger.error(f"     Iteration {i+1}: Failed - {result_normal.get('error', 'Unknown error')}")
                    
            # Calculate statistics
            successful_normal = [r['normal'] for r in test_results if r['normal']['success']]
            successful_stream = [r['streaming'] for r in test_results if r['streaming']['success']]
            
            test_summary = {}
            
            if successful_normal:
                normal_times = [r['response_time'] for r in successful_normal]
                normal_tokens = [r['tokens'] for r in successful_normal]
                
                test_summary['normal'] = {
                    'success_rate': len(successful_normal) / len(test_results),
                    'avg_response_time': statistics.mean(normal_times),
                    'min_response_time': min(normal_times),
                    'max_response_time': max(normal_times),
                    'avg_tokens': statistics.mean(normal_tokens)
                }
                
            if successful_stream:
                stream_times = [r['response_time'] for r in successful_stream]
                stream_tokens = [r['tokens'] for r in successful_stream]
                
                test_summary['streaming'] = {
                    'success_rate': len(successful_stream) / len(test_results),
                    'avg_response_time': statistics.mean(stream_times),
                    'min_response_time': min(stream_times),
                    'max_response_time': max(stream_times),
                    'avg_tokens': statistics.mean(stream_tokens)
                }
                
            benchmark_results['tests'][test_name] = {
                'results': test_results,
                'summary': test_summary
            }
            
        return benchmark_results
        
    def test_api_functionality(self):
        """Test various API functionalities"""
        api_tests = {
            'health_check': False,
            'model_listing': False,
            'chat_completion': False,
            'streaming': False,
            'websocket': False,
            'authentication': False
        }
        
        # Health check
        try:
            response = self.session.get(f"{self.endpoint}/health", timeout=5)
            api_tests['health_check'] = response.status_code == 200
        except:
            pass
            
        # Model listing
        api_tests['model_listing'] = len(self.model_list) > 0
        
        # Chat completion (if models available)
        if self.model_list:
            test_model = self.model_list[0]
            result = self.test_chat_completion(test_model, "Hello", stream=False)
            api_tests['chat_completion'] = result['success']
            
            # Streaming test
            result_stream = self.test_chat_completion(test_model, "Hello", stream=True)
            api_tests['streaming'] = result_stream['success']
            
        # WebSocket test
        ws_result = self.test_websocket_connection()
        api_tests['websocket'] = ws_result['success']
        
        # Authentication test (try accessing a protected endpoint)
        try:
            response = self.session.get(f"{self.endpoint}/api/v1/users", timeout=5)
            api_tests['authentication'] = response.status_code in [200, 401, 403]
        except:
            pass
            
        return api_tests
        
    def create_progress_window(self, total_tests):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("OpenWebUI Testing")
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
        self.status_var = tk.StringVar(value="Initializing OpenWebUI testing...")
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
            output_path = Path.home() / "Desktop" / f"openwebui_test_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        report = {
            'timestamp': timestamp,
            'endpoint': self.endpoint,
            'api_endpoints': self.api_endpoints,
            'available_models': self.model_list,
            'test_results': self.test_results,
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__ if 'torch' in sys.modules else 'Not available'
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
        print("            OPENWEBUI TESTING SUMMARY")
        print("="*70)
        
        print(f"\n🔗 Endpoint: {self.endpoint}")
        
        # API Endpoints
        accessible_endpoints = sum(1 for ep in self.api_endpoints.values() if ep.get('accessible', False))
        print(f"🔌 API Endpoints: {accessible_endpoints}/{len(self.api_endpoints)} accessible")
        
        # Models
        print(f"📋 Available Models: {len(self.model_list)}")
        if self.model_list:
            for i, model in enumerate(self.model_list[:5]):  # Show first 5
                print(f"   • {model}")
            if len(self.model_list) > 5:
                print(f"   ... and {len(self.model_list) - 5} more")
                
        # Test Results
        if 'api_functionality' in self.test_results:
            api_tests = self.test_results['api_functionality']
            passed_tests = sum(1 for result in api_tests.values() if result)
            print(f"\n🧪 API FUNCTIONALITY TESTS: {passed_tests}/{len(api_tests)} passed")
            
            for test_name, result in api_tests.items():
                status = "✅" if result else "❌"
                print(f"   {status} {test_name.replace('_', ' ').title()}")
                
        # Performance Results
        if 'benchmarks' in self.test_results:
            print(f"\n⚡ PERFORMANCE BENCHMARKS:")
            for model_name, benchmark in self.test_results['benchmarks'].items():
                print(f"   📱 Model: {model_name}")
                
                # Show summary stats if available
                for test_name, test_data in benchmark.get('tests', {}).items():
                    if 'summary' in test_data and 'normal' in test_data['summary']:
                        normal_summary = test_data['summary']['normal']
                        success_rate = normal_summary.get('success_rate', 0) * 100
                        avg_time = normal_summary.get('avg_response_time', 0)
                        print(f"      {test_name}: {success_rate:.1f}% success, {avg_time:.2f}s avg")
                        
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
            endpoints_accessible = sum(1 for ep in self.api_endpoints.values() if ep.get('accessible', False))
            
            summary = f"""OpenWebUI Testing Summary

Endpoint: {self.endpoint}
API Endpoints: {endpoints_accessible}/{len(self.api_endpoints)} accessible
Available Models: {models_count}

"""
            
            if 'api_functionality' in self.test_results:
                api_tests = self.test_results['api_functionality']
                passed_tests = sum(1 for result in api_tests.values() if result)
                summary += f"API Tests: {passed_tests}/{len(api_tests)} passed\n"
                
            if 'benchmarks' in self.test_results:
                benchmark_count = len(self.test_results['benchmarks'])
                summary += f"Models Benchmarked: {benchmark_count}\n"
                
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("OpenWebUI Testing Results", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive OpenWebUI Testing Tool")
        parser.add_argument('--endpoint', '-e', default="http://localhost:3000", help='OpenWebUI endpoint URL')
        parser.add_argument('--api-key', '-k', help='API key for authentication')
        parser.add_argument('--model', '-m', help='Specific model to test')
        parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmarks')
        parser.add_argument('--iterations', '-i', type=int, default=3, help='Number of benchmark iterations')
        parser.add_argument('--export', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("OpenWebUI Testing Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.endpoint = args.endpoint
        self.api_key = args.api_key
        
        self.logger.info(f"🚀 Starting OpenWebUI testing on {self.endpoint}...")
        
        try:
            # Setup session
            self.setup_session()
            
            # Check service availability
            if not self.check_service_availability():
                print("❌ OpenWebUI service is not accessible")
                print(f"   Please ensure OpenWebUI is running on {self.endpoint}")
                return
                
            # Discover API endpoints
            print("🔍 Discovering API endpoints...")
            self.discover_api_endpoints()
            
            # Get available models
            print("📋 Getting available models...")
            self.get_available_models()
            
            # Calculate total tests
            total_tests = 1  # API functionality test
            if args.benchmark and self.model_list:
                models_to_test = [args.model] if args.model else self.model_list[:2]  # Test first 2 models
                total_tests += len(models_to_test)
                
            # Create progress window
            self.create_progress_window(total_tests)
            current_test = 0
            
            # Test API functionality
            self.update_progress(current_test, total_tests, "Testing API functionality")
            api_functionality = self.test_api_functionality()
            self.test_results['api_functionality'] = api_functionality
            current_test += 1
            
            # Run benchmarks if requested
            if args.benchmark and self.model_list:
                models_to_test = [args.model] if args.model else self.model_list[:2]
                self.test_results['benchmarks'] = {}
                
                for model_name in models_to_test:
                    self.update_progress(current_test, total_tests, f"Benchmarking {model_name}")
                    benchmark_results = self.benchmark_model_performance(model_name, args.iterations)
                    self.test_results['benchmarks'][model_name] = benchmark_results
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
    tester = OpenWebUITester()
    tester.run()

if __name__ == "__main__":
    main()
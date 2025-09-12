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
# Script Name: gpus-mps-benchmark.py                                             
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive Apple Silicon MPS (Metal Performance Shaders)      
#              benchmarking tool with performance analysis, memory optimization,
#              thermal monitoring, and cross-device comparison capabilities.    
#
# Usage: python gpus-mps-benchmark.py [--tests ALL] [--iterations 10] [--export]
#
# Dependencies: torch, psutil, matplotlib, numpy                               
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional Apple Silicon MPS performance benchmarking with          
#        comprehensive analysis, thermal monitoring, and optimization insights. 
#                                                                                
####################################################################################

"""
Comprehensive Apple Silicon MPS Benchmark Tool
==============================================

Advanced benchmarking tool for Apple Silicon Metal Performance Shaders (MPS)
with performance analysis, memory optimization, thermal monitoring, and
cross-device comparison capabilities for ML development workflows.
"""

import os
import sys
import logging
import subprocess
import argparse
import time
import json
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import statistics
from datetime import datetime

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "torch>=1.12.0",
        "psutil>=5.8.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0"
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
import numpy as np
import psutil
import matplotlib.pyplot as plt

IS_MACOS = sys.platform == "darwin"

class MPSBenchmarkTool:
    def __init__(self):
        self.setup_logging()
        self.device_cpu = torch.device('cpu')
        self.device_mps = None
        self.benchmark_results = {}
        self.system_info = {}
        self.thermal_data = []
        
        # Benchmark configurations
        self.test_configurations = {
            'matrix_small': {'size': 1000, 'iterations': 20},
            'matrix_medium': {'size': 2000, 'iterations': 10},
            'matrix_large': {'size': 4000, 'iterations': 5},
            'vector_ops': {'size': 10000000, 'iterations': 50},
            'conv2d': {'batch': 32, 'channels': 64, 'size': 224, 'iterations': 10},
            'lstm': {'sequence': 100, 'batch': 32, 'hidden': 512, 'iterations': 5},
            'attention': {'sequence': 512, 'batch': 16, 'dim': 512, 'iterations': 5}
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-mps-benchmark.log"
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
        
    def check_mps_availability(self):
        """Check MPS availability and setup devices"""
        self.logger.info("🔍 Checking MPS availability...")
        
        # Check if we're on macOS
        if not IS_MACOS:
            self.logger.error("❌ MPS is only available on macOS")
            return False
            
        # Check PyTorch MPS support
        if not hasattr(torch.backends, 'mps'):
            self.logger.error("❌ PyTorch version doesn't support MPS")
            return False
            
        # Check MPS availability
        if not torch.backends.mps.is_available():
            self.logger.error("❌ MPS is not available on this system")
            return False
            
        # Check MPS build
        if not torch.backends.mps.is_built():
            self.logger.error("❌ PyTorch was not built with MPS support")
            return False
            
        self.device_mps = torch.device('mps')
        self.logger.info("✅ MPS is available and ready")
        return True
        
    def get_system_info(self):
        """Gather comprehensive system information"""
        try:
            # Basic system info
            self.system_info = {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            }
            
            # CPU information
            self.system_info.update({
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            })
            
            # Apple Silicon specific info
            if IS_MACOS:
                try:
                    # Get macOS version details
                    mac_version = platform.mac_ver()
                    self.system_info['macos_version'] = mac_version[0]
                    
                    # Try to get chip information via system_profiler
                    result = subprocess.run(
                        ['system_profiler', 'SPHardwareDataType'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'Chip:' in line:
                                self.system_info['chip'] = line.split('Chip:')[1].strip()
                            elif 'Total Number of Cores:' in line:
                                self.system_info['total_cores'] = line.split(':')[1].strip()
                            elif 'Memory:' in line:
                                self.system_info['total_memory'] = line.split(':')[1].strip()
                                
                except Exception as e:
                    self.logger.warning(f"Could not get detailed system info: {e}")
                    
            return self.system_info
            
        except Exception as e:
            self.logger.error(f"Error gathering system info: {e}")
            return {}
            
    def monitor_thermal_state(self):
        """Monitor thermal state during benchmarks"""
        try:
            # Get CPU temperature if available
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                return {
                                    'sensor': f"{name}_{entry.label}" if entry.label else name,
                                    'temperature': entry.current,
                                    'high': entry.high,
                                    'critical': entry.critical
                                }
            
            # Fallback for macOS - try powermetrics (requires sudo)
            if IS_MACOS:
                try:
                    result = subprocess.run(
                        ['powermetrics', '--sample-count', '1', '-n', '0'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'CPU die temperature' in line:
                                temp = line.split(':')[1].strip().replace('C', '')
                                return {'sensor': 'CPU', 'temperature': float(temp)}
                except:
                    pass
                    
            return {'sensor': 'unavailable', 'temperature': None}
            
        except Exception as e:
            return {'sensor': 'error', 'temperature': None, 'error': str(e)}
            
    def benchmark_matrix_operations(self, size: int, iterations: int, device: torch.device):
        """Benchmark matrix multiplication operations"""
        results = {
            'operation': 'matrix_multiplication',
            'size': size,
            'iterations': iterations,
            'device': str(device),
            'times': [],
            'memory_usage': [],
            'gflops': []
        }
        
        try:
            # Warm-up
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            for _ in range(3):
                _ = torch.matmul(a, b)
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
            # Benchmark
            for i in range(iterations):
                # Monitor memory before operation
                if device.type == 'mps':
                    memory_before = torch.mps.current_allocated_memory()
                else:
                    memory_before = 0
                    
                start_time = time.time()
                c = torch.matmul(a, b)
                
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
                end_time = time.time()
                
                # Calculate metrics
                elapsed_time = end_time - start_time
                gflops = (2 * size**3) / (elapsed_time * 1e9)
                
                if device.type == 'mps':
                    memory_after = torch.mps.current_allocated_memory()
                    memory_used = memory_after - memory_before
                else:
                    memory_used = sys.getsizeof(c.storage())
                    
                results['times'].append(elapsed_time)
                results['gflops'].append(gflops)
                results['memory_usage'].append(memory_used)
                
            # Calculate statistics
            results['avg_time'] = statistics.mean(results['times'])
            results['min_time'] = min(results['times'])
            results['max_time'] = max(results['times'])
            results['std_time'] = statistics.stdev(results['times']) if len(results['times']) > 1 else 0
            results['avg_gflops'] = statistics.mean(results['gflops'])
            results['max_gflops'] = max(results['gflops'])
            
            # Clean up
            del a, b, c
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Matrix benchmark error: {e}")
            
        return results
        
    def benchmark_vector_operations(self, size: int, iterations: int, device: torch.device):
        """Benchmark vector operations"""
        results = {
            'operation': 'vector_operations',
            'size': size,
            'iterations': iterations,
            'device': str(device),
            'operations': {}
        }
        
        try:
            # Create test vectors
            a = torch.randn(size, device=device, dtype=torch.float32)
            b = torch.randn(size, device=device, dtype=torch.float32)
            
            # Test different vector operations
            operations = {
                'add': lambda x, y: x + y,
                'multiply': lambda x, y: x * y,
                'dot_product': lambda x, y: torch.dot(x, y),
                'norm': lambda x, y: torch.norm(x),
                'relu': lambda x, y: torch.relu(x)
            }
            
            for op_name, op_func in operations.items():
                times = []
                
                # Warm-up
                for _ in range(3):
                    _ = op_func(a, b)
                    if device.type == 'mps':
                        torch.mps.synchronize()
                        
                # Benchmark
                for _ in range(iterations):
                    start_time = time.time()
                    result = op_func(a, b)
                    
                    if device.type == 'mps':
                        torch.mps.synchronize()
                        
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                results['operations'][op_name] = {
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'throughput_ops_per_sec': 1.0 / statistics.mean(times)
                }
                
            # Clean up
            del a, b
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Vector benchmark error: {e}")
            
        return results
        
    def benchmark_conv2d_operations(self, batch: int, channels: int, size: int, iterations: int, device: torch.device):
        """Benchmark 2D convolution operations"""
        results = {
            'operation': 'conv2d',
            'batch_size': batch,
            'channels': channels,
            'image_size': size,
            'iterations': iterations,
            'device': str(device),
            'times': []
        }
        
        try:
            # Create test data
            input_tensor = torch.randn(batch, channels, size, size, device=device, dtype=torch.float32)
            
            # Create convolution layer
            conv_layer = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            ).to(device)
            
            # Warm-up
            for _ in range(3):
                _ = conv_layer(input_tensor)
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
            # Benchmark
            for _ in range(iterations):
                start_time = time.time()
                output = conv_layer(input_tensor)
                
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
                end_time = time.time()
                results['times'].append(end_time - start_time)
                
            # Calculate statistics
            results['avg_time'] = statistics.mean(results['times'])
            results['min_time'] = min(results['times'])
            results['max_time'] = max(results['times'])
            results['throughput_samples_per_sec'] = batch / statistics.mean(results['times'])
            
            # Clean up
            del input_tensor, conv_layer, output
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Conv2D benchmark error: {e}")
            
        return results
        
    def benchmark_lstm_operations(self, sequence: int, batch: int, hidden: int, iterations: int, device: torch.device):
        """Benchmark LSTM operations"""
        results = {
            'operation': 'lstm',
            'sequence_length': sequence,
            'batch_size': batch,
            'hidden_size': hidden,
            'iterations': iterations,
            'device': str(device),
            'times': []
        }
        
        try:
            # Create test data
            input_tensor = torch.randn(sequence, batch, hidden, device=device, dtype=torch.float32)
            
            # Create LSTM layer
            lstm_layer = torch.nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=2,
                batch_first=False
            ).to(device)
            
            # Warm-up
            for _ in range(3):
                _ = lstm_layer(input_tensor)
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
            # Benchmark
            for _ in range(iterations):
                start_time = time.time()
                output, (h_n, c_n) = lstm_layer(input_tensor)
                
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
                end_time = time.time()
                results['times'].append(end_time - start_time)
                
            # Calculate statistics
            results['avg_time'] = statistics.mean(results['times'])
            results['min_time'] = min(results['times'])
            results['max_time'] = max(results['times'])
            results['throughput_sequences_per_sec'] = batch / statistics.mean(results['times'])
            
            # Clean up
            del input_tensor, lstm_layer, output
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"LSTM benchmark error: {e}")
            
        return results
        
    def benchmark_attention_operations(self, sequence: int, batch: int, dim: int, iterations: int, device: torch.device):
        """Benchmark attention mechanism operations"""
        results = {
            'operation': 'attention',
            'sequence_length': sequence,
            'batch_size': batch,
            'dimension': dim,
            'iterations': iterations,
            'device': str(device),
            'times': []
        }
        
        try:
            # Create test data
            query = torch.randn(batch, sequence, dim, device=device, dtype=torch.float32)
            key = torch.randn(batch, sequence, dim, device=device, dtype=torch.float32)
            value = torch.randn(batch, sequence, dim, device=device, dtype=torch.float32)
            
            # Attention computation function
            def compute_attention(q, k, v):
                scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
                attention_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attention_weights, v)
                return output
                
            # Warm-up
            for _ in range(3):
                _ = compute_attention(query, key, value)
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
            # Benchmark
            for _ in range(iterations):
                start_time = time.time()
                output = compute_attention(query, key, value)
                
                if device.type == 'mps':
                    torch.mps.synchronize()
                    
                end_time = time.time()
                results['times'].append(end_time - start_time)
                
            # Calculate statistics
            results['avg_time'] = statistics.mean(results['times'])
            results['min_time'] = min(results['times'])
            results['max_time'] = max(results['times'])
            results['throughput_sequences_per_sec'] = batch / statistics.mean(results['times'])
            
            # Clean up
            del query, key, value, output
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"Attention benchmark error: {e}")
            
        return results
        
    def create_progress_window(self, total_tests):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("MPS Benchmarking")
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
        self.status_var = tk.StringVar(value="Initializing MPS benchmarks...")
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
            
    def run_comprehensive_benchmark(self, tests_to_run: List[str]):
        """Run comprehensive benchmark suite"""
        self.logger.info("🚀 Starting comprehensive MPS benchmark...")
        
        devices_to_test = [self.device_cpu]
        if self.device_mps:
            devices_to_test.append(self.device_mps)
            
        total_tests = len(tests_to_run) * len(devices_to_test)
        current_test = 0
        
        # Create progress window
        self.create_progress_window(total_tests)
        
        for device in devices_to_test:
            device_name = str(device)
            self.benchmark_results[device_name] = {}
            
            for test_name in tests_to_run:
                self.update_progress(current_test, total_tests, f"Running {test_name} on {device_name}")
                
                config = self.test_configurations[test_name]
                
                # Record thermal state before test
                thermal_before = self.monitor_thermal_state()
                
                try:
                    if test_name.startswith('matrix'):
                        result = self.benchmark_matrix_operations(
                            config['size'], config['iterations'], device
                        )
                    elif test_name == 'vector_ops':
                        result = self.benchmark_vector_operations(
                            config['size'], config['iterations'], device
                        )
                    elif test_name == 'conv2d':
                        result = self.benchmark_conv2d_operations(
                            config['batch'], config['channels'], config['size'], config['iterations'], device
                        )
                    elif test_name == 'lstm':
                        result = self.benchmark_lstm_operations(
                            config['sequence'], config['batch'], config['hidden'], config['iterations'], device
                        )
                    elif test_name == 'attention':
                        result = self.benchmark_attention_operations(
                            config['sequence'], config['batch'], config['dim'], config['iterations'], device
                        )
                    else:
                        result = {'error': f'Unknown test: {test_name}'}
                        
                    # Record thermal state after test
                    thermal_after = self.monitor_thermal_state()
                    
                    result['thermal_before'] = thermal_before
                    result['thermal_after'] = thermal_after
                    
                    self.benchmark_results[device_name][test_name] = result
                    
                    if 'error' not in result:
                        self.logger.info(f"✅ {test_name} on {device_name}: {result.get('avg_time', 0):.4f}s avg")
                    else:
                        self.logger.error(f"❌ {test_name} on {device_name}: {result['error']}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {test_name} on {device_name} failed: {e}")
                    self.benchmark_results[device_name][test_name] = {'error': str(e)}
                    
                current_test += 1
                
        # Close progress window
        if hasattr(self, 'progress_root'):
            self.progress_root.destroy()
            
        return self.benchmark_results
        
    def generate_performance_comparison(self):
        """Generate performance comparison between CPU and MPS"""
        comparison = {}
        
        if str(self.device_cpu) in self.benchmark_results and str(self.device_mps) in self.benchmark_results:
            cpu_results = self.benchmark_results[str(self.device_cpu)]
            mps_results = self.benchmark_results[str(self.device_mps)]
            
            for test_name in cpu_results.keys():
                if test_name in mps_results:
                    cpu_time = cpu_results[test_name].get('avg_time', None)
                    mps_time = mps_results[test_name].get('avg_time', None)
                    
                    if cpu_time and mps_time and 'error' not in cpu_results[test_name] and 'error' not in mps_results[test_name]:
                        speedup = cpu_time / mps_time
                        comparison[test_name] = {
                            'cpu_time': cpu_time,
                            'mps_time': mps_time,
                            'speedup': speedup,
                            'performance_gain_percent': (speedup - 1) * 100
                        }
                        
        return comparison
        
    def generate_visualization(self, output_dir):
        """Generate performance visualization charts"""
        try:
            comparison = self.generate_performance_comparison()
            
            if not comparison:
                self.logger.warning("No comparison data available for visualization")
                return None
                
            # Create performance comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Apple Silicon MPS vs CPU Performance Comparison', fontsize=16)
            
            # Speedup chart
            test_names = list(comparison.keys())
            speedups = [comparison[test]['speedup'] for test in test_names]
            
            bars1 = ax1.bar(test_names, speedups, color='skyblue', alpha=0.7)
            ax1.set_ylabel('Speedup Factor (MPS vs CPU)')
            ax1.set_title('Performance Speedup')
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax1.legend()
            
            # Add value labels on bars
            for bar, speedup in zip(bars1, speedups):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{speedup:.1f}x', ha='center', va='bottom')
                        
            # Execution time comparison
            cpu_times = [comparison[test]['cpu_time'] * 1000 for test in test_names]  # Convert to ms
            mps_times = [comparison[test]['mps_time'] * 1000 for test in test_names]  # Convert to ms
            
            x = np.arange(len(test_names))
            width = 0.35
            
            bars2 = ax2.bar(x - width/2, cpu_times, width, label='CPU', color='lightcoral', alpha=0.7)
            bars3 = ax2.bar(x + width/2, mps_times, width, label='MPS', color='lightgreen', alpha=0.7)
            
            ax2.set_ylabel('Execution Time (ms)')
            ax2.set_title('Execution Time Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(test_names, rotation=45, ha='right')
            ax2.legend()
            ax2.set_yscale('log')  # Log scale for better visualization
            
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f"mps_performance_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Performance chart saved to: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            return None
            
    def generate_report(self, output_path=None):
        """Generate comprehensive benchmark report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_dir = Path.home() / "Desktop"
            output_path = output_dir / f"mps_benchmark_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            output_dir = output_path.parent
            
        # Generate performance comparison
        comparison = self.generate_performance_comparison()
        
        # Generate visualization
        chart_path = self.generate_visualization(output_dir)
        
        report = {
            'timestamp': timestamp,
            'system_info': self.system_info,
            'benchmark_results': self.benchmark_results,
            'performance_comparison': comparison,
            'chart_generated': str(chart_path) if chart_path else None,
            'test_configurations': self.test_configurations
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
        """Display benchmark results summary"""
        print("\n" + "="*70)
        print("            APPLE SILICON MPS BENCHMARK SUMMARY")
        print("="*70)
        
        # System Information
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   Platform: {self.system_info.get('platform', 'Unknown')}")
        print(f"   Chip: {self.system_info.get('chip', 'Unknown')}")
        print(f"   macOS: {self.system_info.get('macos_version', 'Unknown')}")
        print(f"   PyTorch: {self.system_info.get('torch_version', 'Unknown')}")
        print(f"   Memory: {self.system_info.get('memory_total_gb', 0)} GB")
        print(f"   MPS Available: {'✅ Yes' if self.system_info.get('mps_available') else '❌ No'}")
        
        # Performance Comparison
        comparison = self.generate_performance_comparison()
        if comparison:
            print(f"\n⚡ PERFORMANCE COMPARISON (MPS vs CPU):")
            print(f"   {'Test':<15} {'CPU Time':<12} {'MPS Time':<12} {'Speedup':<10}")
            print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
            
            for test_name, data in comparison.items():
                cpu_time = data['cpu_time'] * 1000  # Convert to ms
                mps_time = data['mps_time'] * 1000  # Convert to ms
                speedup = data['speedup']
                
                print(f"   {test_name:<15} {cpu_time:>8.2f} ms {mps_time:>8.2f} ms {speedup:>6.1f}x")
                
            # Overall summary
            avg_speedup = statistics.mean([data['speedup'] for data in comparison.values()])
            print(f"\n   Average Speedup: {avg_speedup:.1f}x")
            
        # Best performing operations
        if self.device_mps and str(self.device_mps) in self.benchmark_results:
            mps_results = self.benchmark_results[str(self.device_mps)]
            print(f"\n🏆 MPS PERFORMANCE HIGHLIGHTS:")
            
            # Find operations with best GFLOPS
            gflops_tests = [(name, result.get('avg_gflops', 0)) for name, result in mps_results.items() 
                           if 'avg_gflops' in result and 'error' not in result]
            
            if gflops_tests:
                gflops_tests.sort(key=lambda x: x[1], reverse=True)
                best_gflops = gflops_tests[0]
                print(f"   Best GFLOPS: {best_gflops[0]} ({best_gflops[1]:.1f} GFLOPS)")
                
        print("\n" + "="*70)
        
    def show_gui_summary(self):
        """Show GUI summary of benchmark results"""
        if not IS_MACOS:
            return
            
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Create summary message
            mps_available = self.system_info.get('mps_available', False)
            chip = self.system_info.get('chip', 'Unknown')
            
            summary = f"""Apple Silicon MPS Benchmark Summary

System: {chip}
MPS Available: {'✅ Yes' if mps_available else '❌ No'}

"""
            
            if mps_available:
                comparison = self.generate_performance_comparison()
                if comparison:
                    avg_speedup = statistics.mean([data['speedup'] for data in comparison.values()])
                    summary += f"Average MPS Speedup: {avg_speedup:.1f}x\n\n"
                    
                    # Show top 3 speedups
                    sorted_tests = sorted(comparison.items(), key=lambda x: x[1]['speedup'], reverse=True)
                    summary += "Top Performance Gains:\n"
                    for test_name, data in sorted_tests[:3]:
                        summary += f"• {test_name}: {data['speedup']:.1f}x faster\n"
                        
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("MPS Benchmark Results", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive Apple Silicon MPS Benchmark Tool")
        parser.add_argument('--tests', '-t', default='all', help='Tests to run (comma-separated or "all")')
        parser.add_argument('--iterations', '-i', type=int, help='Override default iterations')
        parser.add_argument('--export', '-e', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--chart', '-c', action='store_true', help='Generate performance charts')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Apple Silicon MPS Benchmark Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("🚀 Starting Apple Silicon MPS benchmarking...")
        
        try:
            # Get system information
            self.get_system_info()
            
            # Check MPS availability
            if not self.check_mps_availability():
                print("❌ MPS is not available. CPU benchmarking only.")
                
            # Determine tests to run
            if args.tests.lower() == 'all':
                tests_to_run = list(self.test_configurations.keys())
            else:
                tests_to_run = [t.strip() for t in args.tests.split(',')]
                # Validate test names
                invalid_tests = [t for t in tests_to_run if t not in self.test_configurations]
                if invalid_tests:
                    print(f"❌ Invalid test names: {invalid_tests}")
                    print(f"Available tests: {list(self.test_configurations.keys())}")
                    return
                    
            # Override iterations if specified
            if args.iterations:
                for config in self.test_configurations.values():
                    config['iterations'] = args.iterations
                    
            # Run benchmarks
            self.run_comprehensive_benchmark(tests_to_run)
            
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
            self.logger.info("Benchmarking interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

def main():
    benchmark_tool = MPSBenchmarkTool()
    benchmark_tool.run()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

PyTorch Performance Benchmark Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive PyTorch performance benchmarking and optimization analysis

Enhanced with GET SWIFTY - Universal macOS compatibility and professional features.
"""

import sys
import os
import subprocess
import json
import time
import platform
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'torch',
        'torchvision', 
        'psutil',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    for package in dependencies:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade", "--quiet"
                ])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                return False
    return True

# Install dependencies before importing
if not install_dependencies():
    print("Failed to install required dependencies. Exiting.")
    sys.exit(1)

import torch
import torchvision
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PyTorchPerformanceBenchmark:
    """Comprehensive PyTorch performance benchmarking suite."""
    
    def __init__(self):
        """Initialize the PyTorch performance benchmark suite."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"pytorch_performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Benchmark configuration
        self.warmup_iterations = 5
        self.benchmark_iterations = 20
        self.test_sizes = [256, 512, 1024, 2048, 4096]
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"PyTorch Performance Benchmark Log - {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
    
    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and optionally print to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass  # Fail silently for logging issues
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def get_available_devices(self) -> Dict[str, Any]:
        """Get all available devices for computation."""
        self._log("Detecting available devices")
        
        devices = {
            'cpu': {
                'available': True,
                'device': torch.device('cpu'),
                'info': f"CPU ({self.platform_info['cpu_count']} cores)",
                'memory_gb': self.platform_info['memory_gb']
            }
        }
        
        # Check CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = f"cuda:{i}"
                device_info = torch.cuda.get_device_name(i)
                properties = torch.cuda.get_device_properties(i)
                
                devices[device_name] = {
                    'available': True,
                    'device': torch.device(device_name),
                    'info': device_info,
                    'memory_gb': round(properties.total_memory / (1024**3), 2),
                    'properties': {
                        'total_memory': properties.total_memory,
                        'multiprocessor_count': properties.multi_processor_count,
                        'compute_capability': f"{properties.major}.{properties.minor}",
                        'max_threads_per_multiprocessor': properties.max_threads_per_multiprocessor,
                        'max_shared_memory_per_multiprocessor': properties.max_shared_memory_per_multiprocessor
                    }
                }
        
        # Check MPS device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices['mps'] = {
                'available': True,
                'device': torch.device('mps'),
                'info': "Apple Metal Performance Shaders",
                'memory_gb': self.platform_info['memory_gb']  # Shared with system memory
            }
        
        self._log(f"Found {len(devices)} available devices")
        return devices
    
    def benchmark_matrix_operations(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark matrix operations on specified device."""
        self._log(f"Benchmarking matrix operations on {device}")
        
        results = {
            'device': str(device),
            'operations': {}
        }
        
        operations = [
            ('matrix_multiplication', self._benchmark_matmul),
            ('element_wise_operations', self._benchmark_elementwise),
            ('linear_algebra', self._benchmark_linalg),
            ('convolution_operations', self._benchmark_convolution),
            ('fft_operations', self._benchmark_fft),
            ('reduction_operations', self._benchmark_reductions)
        ]
        
        for op_name, op_func in operations:
            self._log(f"Benchmarking {op_name}")
            
            try:
                operation_results = {}
                
                for size in self.test_sizes:
                    if self._skip_size_for_device(device, size):
                        continue
                    
                    self._log(f"Testing {op_name} with size {size}")
                    
                    # Run warmup
                    for _ in range(self.warmup_iterations):
                        try:
                            _ = op_func(device, size)
                        except Exception:
                            break
                    
                    # Benchmark
                    times = []
                    for _ in range(self.benchmark_iterations):
                        try:
                            exec_time = op_func(device, size)
                            times.append(exec_time)
                        except Exception as e:
                            self._log(f"Error in {op_name} size {size}: {e}", "WARNING")
                            break
                    
                    if times:
                        operation_results[str(size)] = {
                            'mean_time': statistics.mean(times),
                            'median_time': statistics.median(times),
                            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                            'min_time': min(times),
                            'max_time': max(times),
                            'samples': len(times),
                            'throughput_gflops': self._calculate_gflops(op_name, size, statistics.mean(times))
                        }
                
                results['operations'][op_name] = operation_results
                self._log(f"✓ {op_name} benchmark completed")
                
            except Exception as e:
                self._log(f"✗ {op_name} benchmark failed: {e}", "ERROR")
                results['operations'][op_name] = {'error': str(e)}
        
        return results
    
    def _skip_size_for_device(self, device: torch.device, size: int) -> bool:
        """Determine if a size should be skipped for a device."""
        # Skip very large sizes for CPU to avoid excessive runtime
        if device.type == 'cpu' and size > 2048:
            return True
        
        # Skip very large sizes for devices with limited memory
        if device.type == 'mps' and size > 4096:
            return True
        
        return False
    
    def _calculate_gflops(self, operation: str, size: int, time_seconds: float) -> float:
        """Calculate GFLOPS for different operations."""
        if time_seconds <= 0:
            return 0.0
        
        if operation == 'matrix_multiplication':
            # Matrix multiplication: 2 * n^3 operations
            operations = 2 * size**3
        elif operation == 'element_wise_operations':
            # Element-wise operations: n^2 operations
            operations = size**2
        elif operation == 'linear_algebra':
            # SVD/eigenvalue decomposition: approximately O(n^3)
            operations = size**3
        elif operation == 'convolution_operations':
            # Convolution: depends on kernel size, estimate
            operations = size**2 * 9  # 3x3 kernel estimate
        elif operation == 'fft_operations':
            # FFT: n * log(n) operations
            operations = size**2 * np.log2(size**2)
        elif operation == 'reduction_operations':
            # Reduction: n^2 operations
            operations = size**2
        else:
            operations = size**2
        
        return operations / (time_seconds * 1e9)
    
    def _benchmark_matmul(self, device: torch.device, size: int) -> float:
        """Benchmark matrix multiplication."""
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        c = torch.matmul(a, b)
        
        # Synchronize after operation
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del a, b, c
        
        return end_time - start_time
    
    def _benchmark_elementwise(self, device: torch.device, size: int) -> float:
        """Benchmark element-wise operations."""
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Perform multiple element-wise operations
        c = a + b
        d = c * a
        e = torch.sin(d)
        f = torch.exp(e.clamp(-10, 10))  # Clamp to avoid overflow
        result = torch.sum(f)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del a, b, c, d, e, f, result
        
        return end_time - start_time
    
    def _benchmark_linalg(self, device: torch.device, size: int) -> float:
        """Benchmark linear algebra operations."""
        # Use smaller size for linear algebra to avoid memory issues
        test_size = min(size, 1024)
        a = torch.randn(test_size, test_size, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        try:
            # SVD decomposition
            u, s, v = torch.svd(a)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            # Clean up
            del u, s, v
            
        except Exception:
            # Fallback to simpler operation
            result = torch.det(a)
            del result
        
        end_time = time.time()
        
        # Clean up
        del a
        
        return end_time - start_time
    
    def _benchmark_convolution(self, device: torch.device, size: int) -> float:
        """Benchmark convolution operations."""
        # Create input tensor (batch, channels, height, width)
        input_tensor = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
        
        # Create convolution layer
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        output = conv(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del input_tensor, output, conv
        
        return end_time - start_time
    
    def _benchmark_fft(self, device: torch.device, size: int) -> float:
        """Benchmark FFT operations."""
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # 2D FFT
        fft_result = torch.fft.fft2(a)
        ifft_result = torch.fft.ifft2(fft_result)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del a, fft_result, ifft_result
        
        return end_time - start_time
    
    def _benchmark_reductions(self, device: torch.device, size: int) -> float:
        """Benchmark reduction operations."""
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Various reduction operations
        sum_result = torch.sum(a)
        mean_result = torch.mean(a)
        max_result = torch.max(a)
        min_result = torch.min(a)
        std_result = torch.std(a)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del a, sum_result, mean_result, max_result, min_result, std_result
        
        return end_time - start_time
    
    def benchmark_neural_network_training(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark neural network training performance."""
        self._log(f"Benchmarking neural network training on {device}")
        
        results = {
            'device': str(device),
            'network_benchmarks': {}
        }
        
        networks = [
            ('simple_mlp', self._benchmark_mlp_training),
            ('convolutional_net', self._benchmark_cnn_training),
            ('transformer_block', self._benchmark_transformer_training)
        ]
        
        for net_name, net_func in networks:
            self._log(f"Benchmarking {net_name}")
            
            try:
                net_results = {}
                
                # Test with different batch sizes
                batch_sizes = [16, 32, 64] if device.type != 'cpu' else [16, 32]
                
                for batch_size in batch_sizes:
                    self._log(f"Testing {net_name} with batch size {batch_size}")
                    
                    try:
                        # Warmup
                        for _ in range(3):
                            _ = net_func(device, batch_size)
                        
                        # Benchmark
                        times = []
                        for _ in range(10):
                            exec_time = net_func(device, batch_size)
                            times.append(exec_time)
                        
                        net_results[f'batch_{batch_size}'] = {
                            'mean_time': statistics.mean(times),
                            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                            'throughput_samples_per_sec': batch_size / statistics.mean(times)
                        }
                        
                    except Exception as e:
                        self._log(f"Error in {net_name} batch {batch_size}: {e}", "WARNING")
                        net_results[f'batch_{batch_size}'] = {'error': str(e)}
                
                results['network_benchmarks'][net_name] = net_results
                self._log(f"✓ {net_name} benchmark completed")
                
            except Exception as e:
                self._log(f"✗ {net_name} benchmark failed: {e}", "ERROR")
                results['network_benchmarks'][net_name] = {'error': str(e)}
        
        return results
    
    def _benchmark_mlp_training(self, device: torch.device, batch_size: int) -> float:
        """Benchmark MLP training step."""
        # Simple MLP
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Generate random data
        x = torch.randn(batch_size, 784, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del model, optimizer, criterion, x, y, output, loss
        
        return end_time - start_time
    
    def _benchmark_cnn_training(self, device: torch.device, batch_size: int) -> float:
        """Benchmark CNN training step."""
        # Simple CNN
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Generate random data (CIFAR-10 like)
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del model, optimizer, criterion, x, y, output, loss
        
        return end_time - start_time
    
    def _benchmark_transformer_training(self, device: torch.device, batch_size: int) -> float:
        """Benchmark Transformer block training step."""
        # Simple transformer block
        d_model = 512
        nhead = 8
        
        model = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Generate random sequence data
        seq_len = 64
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        # Use mean pooling to reduce sequence to single vector
        transformer_out = model[0](x)
        pooled = torch.mean(transformer_out, dim=1)
        output = model[1:](pooled)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        # Clean up
        del model, optimizer, criterion, x, y, transformer_out, pooled, output, loss
        
        return end_time - start_time
    
    def compare_device_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across devices."""
        self._log("Comparing device performance")
        
        comparison = {
            'device_rankings': {},
            'operation_speedups': {},
            'best_device_per_operation': {}
        }
        
        # Collect all operations across devices
        all_operations = set()
        for device_result in results.values():
            if 'matrix_operations' in device_result:
                all_operations.update(device_result['matrix_operations']['operations'].keys())
        
        # Compare each operation
        for operation in all_operations:
            operation_times = {}
            
            for device_name, device_result in results.items():
                if ('matrix_operations' in device_result and 
                    operation in device_result['matrix_operations']['operations']):
                    
                    op_data = device_result['matrix_operations']['operations'][operation]
                    if isinstance(op_data, dict) and 'error' not in op_data:
                        # Use largest available size for comparison
                        available_sizes = [k for k in op_data.keys() if k.isdigit()]
                        if available_sizes:
                            largest_size = max(available_sizes, key=int)
                            if 'mean_time' in op_data[largest_size]:
                                operation_times[device_name] = op_data[largest_size]['mean_time']
            
            if len(operation_times) > 1:
                # Find best (fastest) device
                best_device = min(operation_times.keys(), key=lambda x: operation_times[x])
                best_time = operation_times[best_device]
                
                comparison['best_device_per_operation'][operation] = best_device
                
                # Calculate speedups relative to CPU (if available)
                if 'cpu' in operation_times:
                    cpu_time = operation_times['cpu']
                    speedups = {}
                    for device, time_val in operation_times.items():
                        if device != 'cpu':
                            speedups[device] = cpu_time / time_val
                    comparison['operation_speedups'][operation] = speedups
        
        return comparison
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance benchmark report."""
        self._log("Generating comprehensive performance report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'pytorch_version': torch.__version__,
                'log_file': str(self.log_file)
            },
            'devices': self.get_available_devices(),
            'benchmark_config': {
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations,
                'test_sizes': self.test_sizes
            },
            'device_results': {},
            'performance_comparison': {},
            'summary': {}
        }
        
        # Test all available devices
        devices = report['devices']
        
        for device_name, device_info in devices.items():
            if device_info['available']:
                device = device_info['device']
                self._log(f"Running benchmarks on device: {device_name}")
                
                try:
                    # Run all benchmarks
                    report['device_results'][device_name] = {
                        'matrix_operations': self.benchmark_matrix_operations(device),
                        'neural_network_training': self.benchmark_neural_network_training(device)
                    }
                    
                except Exception as e:
                    self._log(f"Benchmark failed for {device_name}: {e}", "ERROR")
                    report['device_results'][device_name] = {
                        'error': str(e)
                    }
        
        # Generate performance comparison
        report['performance_comparison'] = self.compare_device_performance(report['device_results'])
        
        # Generate summary
        report['summary'] = self._generate_performance_summary(report)
        
        return report
    
    def _generate_performance_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of performance benchmark results."""
        summary = {
            'devices_tested': len(report['device_results']),
            'successful_tests': 0,
            'failed_tests': 0,
            'performance_highlights': {},
            'recommendations': []
        }
        
        for device_name, device_results in report['device_results'].items():
            if 'error' not in device_results:
                summary['successful_tests'] += 1
                
                # Extract performance highlights
                highlights = {}
                
                if 'matrix_operations' in device_results:
                    matrix_ops = device_results['matrix_operations']['operations']
                    max_gflops = 0
                    best_operation = None
                    
                    for op_name, op_data in matrix_ops.items():
                        if isinstance(op_data, dict) and 'error' not in op_data:
                            for size_data in op_data.values():
                                if isinstance(size_data, dict) and 'throughput_gflops' in size_data:
                                    gflops = size_data['throughput_gflops']
                                    if gflops > max_gflops:
                                        max_gflops = gflops
                                        best_operation = op_name
                    
                    highlights['max_gflops'] = max_gflops
                    highlights['best_operation'] = best_operation
                
                if 'neural_network_training' in device_results:
                    nn_results = device_results['neural_network_training']['network_benchmarks']
                    max_throughput = 0
                    
                    for net_name, net_data in nn_results.items():
                        if isinstance(net_data, dict):
                            for batch_data in net_data.values():
                                if isinstance(batch_data, dict) and 'throughput_samples_per_sec' in batch_data:
                                    throughput = batch_data['throughput_samples_per_sec']
                                    max_throughput = max(max_throughput, throughput)
                    
                    highlights['max_training_throughput'] = max_throughput
                
                summary['performance_highlights'][device_name] = highlights
                
            else:
                summary['failed_tests'] += 1
        
        # Generate recommendations
        if summary['failed_tests'] > 0:
            summary['recommendations'].append("Some benchmark tests failed. Check available memory and reduce test sizes if needed.")
        
        if 'cuda:0' in report['devices'] and report['devices']['cuda:0']['available']:
            summary['recommendations'].append("CUDA GPU detected. Use GPU acceleration for matrix operations and neural network training.")
        
        if 'mps' in report['devices'] and report['devices']['mps']['available']:
            summary['recommendations'].append("Apple Silicon MPS detected. Optimal for neural network training on Apple Silicon Macs.")
        
        # Performance comparison recommendations
        if 'operation_speedups' in report['performance_comparison']:
            max_speedup = 0
            best_gpu = None
            
            for operation, speedups in report['performance_comparison']['operation_speedups'].items():
                for gpu, speedup in speedups.items():
                    if speedup > max_speedup:
                        max_speedup = speedup
                        best_gpu = gpu
            
            if max_speedup > 1:
                summary['recommendations'].append(f"Use {best_gpu} for best performance - up to {max_speedup:.1f}x speedup over CPU.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"pytorch_performance_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create performance benchmark visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('PyTorch Performance Benchmark Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Matrix Multiplication Performance
            ax1 = axes[0, 0]
            for device_name, device_results in results['device_results'].items():
                if ('matrix_operations' in device_results and 
                    'matrix_multiplication' in device_results['matrix_operations']['operations']):
                    
                    matmul_data = device_results['matrix_operations']['operations']['matrix_multiplication']
                    sizes = []
                    gflops = []
                    
                    for size_str, size_data in matmul_data.items():
                        if size_str.isdigit() and isinstance(size_data, dict) and 'throughput_gflops' in size_data:
                            sizes.append(int(size_str))
                            gflops.append(size_data['throughput_gflops'])
                    
                    if sizes:
                        ax1.plot(sizes, gflops, marker='o', label=device_name, linewidth=2)
            
            ax1.set_xlabel('Matrix Size')
            ax1.set_ylabel('Performance (GFLOPS)')
            ax1.set_title('Matrix Multiplication Performance')
            ax1.set_xscale('log', base=2)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Training Throughput Comparison
            ax2 = axes[0, 1]
            device_names = []
            throughputs = []
            
            for device_name, device_results in results['device_results'].items():
                if 'neural_network_training' in device_results:
                    nn_data = device_results['neural_network_training']['network_benchmarks']
                    max_throughput = 0
                    
                    for net_data in nn_data.values():
                        if isinstance(net_data, dict):
                            for batch_data in net_data.values():
                                if isinstance(batch_data, dict) and 'throughput_samples_per_sec' in batch_data:
                                    max_throughput = max(max_throughput, batch_data['throughput_samples_per_sec'])
                    
                    if max_throughput > 0:
                        device_names.append(device_name)
                        throughputs.append(max_throughput)
            
            if device_names:
                bars = ax2.bar(device_names, throughputs, color=plt.cm.viridis(np.linspace(0, 1, len(device_names))))
                ax2.set_ylabel('Samples/Second')
                ax2.set_title('Neural Network Training Throughput')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, throughputs):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # Plot 3: Speedup Over CPU
            ax3 = axes[0, 2]
            if 'operation_speedups' in results['performance_comparison']:
                operations = list(results['performance_comparison']['operation_speedups'].keys())
                speedup_data = results['performance_comparison']['operation_speedups']
                
                devices = set()
                for op_speedups in speedup_data.values():
                    devices.update(op_speedups.keys())
                devices = list(devices)
                
                x_pos = np.arange(len(operations))
                width = 0.35
                
                for i, device in enumerate(devices):
                    speedups = []
                    for operation in operations:
                        speedup = speedup_data.get(operation, {}).get(device, 0)
                        speedups.append(speedup)
                    
                    ax3.bar(x_pos + i*width, speedups, width, label=device, alpha=0.8)
                
                ax3.set_xlabel('Operations')
                ax3.set_ylabel('Speedup over CPU')
                ax3.set_title('GPU Speedup Comparison')
                ax3.set_xticks(x_pos + width/2)
                ax3.set_xticklabels([op.replace('_', ' ').title() for op in operations], rotation=45)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Memory vs Performance Trade-off
            ax4 = axes[1, 0]
            for device_name, device_info in results['devices'].items():
                if device_info['available'] and device_name in results['device_results']:
                    memory_gb = device_info.get('memory_gb', 0)
                    
                    # Get peak performance
                    peak_gflops = 0
                    device_results = results['device_results'][device_name]
                    if 'matrix_operations' in device_results:
                        for op_data in device_results['matrix_operations']['operations'].values():
                            if isinstance(op_data, dict):
                                for size_data in op_data.values():
                                    if isinstance(size_data, dict) and 'throughput_gflops' in size_data:
                                        peak_gflops = max(peak_gflops, size_data['throughput_gflops'])
                    
                    if peak_gflops > 0 and memory_gb > 0:
                        ax4.scatter(memory_gb, peak_gflops, s=100, label=device_name, alpha=0.7)
                        ax4.annotate(device_name, (memory_gb, peak_gflops), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax4.set_xlabel('Memory (GB)')
            ax4.set_ylabel('Peak Performance (GFLOPS)')
            ax4.set_title('Memory vs Performance')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Operation Comparison Radar Chart
            ax5 = axes[1, 1]
            if len(results['device_results']) > 1:
                operations = []
                device_data = {}
                
                # Collect normalized performance data
                for device_name, device_results in results['device_results'].items():
                    if 'matrix_operations' in device_results:
                        device_data[device_name] = {}
                        for op_name, op_data in device_results['matrix_operations']['operations'].items():
                            if isinstance(op_data, dict) and 'error' not in op_data:
                                # Get best performance for this operation
                                max_gflops = 0
                                for size_data in op_data.values():
                                    if isinstance(size_data, dict) and 'throughput_gflops' in size_data:
                                        max_gflops = max(max_gflops, size_data['throughput_gflops'])
                                device_data[device_name][op_name] = max_gflops
                                if op_name not in operations:
                                    operations.append(op_name)
                
                # Create comparison chart
                if operations and device_data:
                    x_pos = np.arange(len(operations))
                    width = 0.8 / len(device_data)
                    
                    for i, (device_name, perf_data) in enumerate(device_data.items()):
                        values = [perf_data.get(op, 0) for op in operations]
                        ax5.bar(x_pos + i*width, values, width, label=device_name, alpha=0.7)
                    
                    ax5.set_xlabel('Operations')
                    ax5.set_ylabel('Performance (GFLOPS)')
                    ax5.set_title('Operation Performance Comparison')
                    ax5.set_xticks(x_pos + width/2)
                    ax5.set_xticklabels([op.replace('_', ' ').title() for op in operations], rotation=45)
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
            
            # Plot 6: Performance Summary
            ax6 = axes[1, 2]
            summary_data = results['summary']['performance_highlights']
            devices = list(summary_data.keys())
            
            if devices:
                max_gflops = [summary_data[dev].get('max_gflops', 0) for dev in devices]
                max_training = [summary_data[dev].get('max_training_throughput', 0) for dev in devices]
                
                x_pos = np.arange(len(devices))
                width = 0.35
                
                bars1 = ax6.bar(x_pos - width/2, max_gflops, width, label='Peak GFLOPS', alpha=0.8)
                
                # Secondary y-axis for training throughput
                ax6_twin = ax6.twinx()
                bars2 = ax6_twin.bar(x_pos + width/2, max_training, width, 
                                   label='Training Throughput', alpha=0.8, color='orange')
                
                ax6.set_xlabel('Devices')
                ax6.set_ylabel('Peak GFLOPS', color='blue')
                ax6_twin.set_ylabel('Samples/Second', color='orange')
                ax6.set_title('Performance Summary')
                ax6.set_xticks(x_pos)
                ax6.set_xticklabels(devices, rotation=45)
                
                # Combined legend
                lines1, labels1 = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6_twin.get_legend_handles_labels()
                ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"pytorch_performance_benchmark_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Visualization saved to: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            self._log(f"Failed to create visualizations: {e}", "ERROR")
            return ""
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n" + "="*80)
        print("PYTORCH PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"PyTorch: {results['metadata']['pytorch_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"System Memory: {platform_info['memory_gb']} GB")
        
        # Device summary
        print(f"\nDevices Tested: {results['summary']['devices_tested']}")
        print(f"Successful Tests: {results['summary']['successful_tests']}")
        print(f"Failed Tests: {results['summary']['failed_tests']}")
        
        # Performance highlights
        print(f"\nPerformance Highlights:")
        for device_name, highlights in results['summary']['performance_highlights'].items():
            print(f"  {device_name}:")
            if 'max_gflops' in highlights and highlights['max_gflops'] > 0:
                print(f"    Peak Performance: {highlights['max_gflops']:.2f} GFLOPS ({highlights.get('best_operation', 'unknown')})")
            if 'max_training_throughput' in highlights and highlights['max_training_throughput'] > 0:
                print(f"    Training Throughput: {highlights['max_training_throughput']:.1f} samples/sec")
        
        # Best devices per operation
        if 'best_device_per_operation' in results['performance_comparison']:
            print(f"\nBest Device Per Operation:")
            for operation, best_device in results['performance_comparison']['best_device_per_operation'].items():
                print(f"  {operation.replace('_', ' ').title()}: {best_device}")
        
        # Speedup information
        if 'operation_speedups' in results['performance_comparison']:
            print(f"\nGPU Speedups over CPU:")
            for operation, speedups in results['performance_comparison']['operation_speedups'].items():
                print(f"  {operation.replace('_', ' ').title()}:")
                for device, speedup in speedups.items():
                    print(f"    {device}: {speedup:.1f}x")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run PyTorch performance benchmark."""
    print("Starting PyTorch Performance Benchmark Suite...")
    
    try:
        # Initialize benchmark
        benchmark = PyTorchPerformanceBenchmark()
        
        # Run comprehensive performance benchmarks
        results = benchmark.generate_performance_report()
        
        # Save results
        results_file = benchmark.save_results(results)
        
        # Create visualizations
        viz_file = benchmark.create_visualizations(results)
        
        # Print summary
        benchmark.print_summary(results)
        
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        if viz_file:
            print(f"Visualizations saved to: {viz_file}")
        
        # Show macOS notification if available
        try:
            if platform.system() == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    f'display notification "PyTorch performance benchmark completed successfully!" with title "GET SWIFTY - Performance Benchmark"'
                ], check=False)
        except:
            pass
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
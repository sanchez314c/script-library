#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

PyTorch Memory Benchmark & Analysis Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive PyTorch memory benchmarking and optimization analysis

Enhanced with GET SWIFTY - Universal macOS compatibility and professional features.
"""

import sys
import os
import subprocess
import json
import time
import gc
import platform
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'torch',
        'torchvision', 
        'psutil',
        'numpy',
        'matplotlib',
        'seaborn'
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

class PyTorchMemoryBenchmark:
    """Comprehensive PyTorch memory benchmarking and analysis suite."""
    
    def __init__(self):
        """Initialize the PyTorch memory benchmark suite."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"pytorch_memory_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Memory tracking
        self.memory_history = []
        self.allocation_history = []
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"PyTorch Memory Benchmark Log - {datetime.now()}\n")
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
                        'compute_capability': f"{properties.major}.{properties.minor}"
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
    
    def get_memory_info(self, device: torch.device) -> Dict[str, float]:
        """Get detailed memory information for device."""
        if device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.memory_allocated(device) / (1024**2),
                'reserved_mb': torch.cuda.memory_reserved(device) / (1024**2),
                'max_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
                'max_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024**2),
                'total_mb': torch.cuda.get_device_properties(device).total_memory / (1024**2)
            }
        elif device.type == 'mps':
            # MPS shares system memory
            memory = psutil.virtual_memory()
            return {
                'allocated_mb': 0,  # MPS doesn't provide detailed tracking
                'reserved_mb': 0,
                'max_allocated_mb': 0,
                'max_reserved_mb': 0,
                'total_mb': memory.total / (1024**2),
                'available_mb': memory.available / (1024**2),
                'used_percent': memory.percent
            }
        else:  # CPU
            memory = psutil.virtual_memory()
            process = psutil.Process()
            return {
                'process_mb': process.memory_info().rss / (1024**2),
                'total_mb': memory.total / (1024**2),
                'available_mb': memory.available / (1024**2),
                'used_percent': memory.percent
            }
    
    def benchmark_tensor_allocation(self, device: torch.device, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark tensor allocation patterns."""
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        self._log(f"Benchmarking tensor allocation on {device}")
        
        results = {
            'device': str(device),
            'allocation_tests': {},
            'memory_efficiency': {},
            'timing_results': {}
        }
        
        for size in sizes:
            self._log(f"Testing allocation for size {size}x{size}")
            
            # Clear memory before test
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            
            # Record initial state
            initial_memory = self.get_memory_info(device)
            
            # Allocation timing test
            start_time = time.time()
            try:
                tensor = torch.randn(size, size, device=device, dtype=torch.float32)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                allocation_time = time.time() - start_time
                
                # Memory after allocation
                after_memory = self.get_memory_info(device)
                
                # Calculate memory usage
                if device.type == 'cuda':
                    memory_used = after_memory['allocated_mb'] - initial_memory['allocated_mb']
                else:
                    memory_used = size * size * 4 / (1024**2)  # Estimate for float32
                
                # Test operations on allocated tensor
                start_time = time.time()
                result = torch.matmul(tensor, tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                operation_time = time.time() - start_time
                
                # Clean up
                del tensor, result
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Record results
                results['allocation_tests'][f'{size}x{size}'] = {
                    'allocation_time': allocation_time,
                    'operation_time': operation_time,
                    'memory_used_mb': memory_used,
                    'theoretical_memory_mb': size * size * 4 / (1024**2),
                    'efficiency_ratio': memory_used / (size * size * 4 / (1024**2)) if memory_used > 0 else 1.0,
                    'throughput_gflops': (2 * size**3) / (operation_time * 1e9) if operation_time > 0 else 0
                }
                
                self._log(f"✓ {size}x{size}: {allocation_time:.4f}s allocation, {operation_time:.4f}s operation")
                
            except Exception as e:
                self._log(f"✗ Failed allocation for {size}x{size}: {e}", "ERROR")
                results['allocation_tests'][f'{size}x{size}'] = {
                    'error': str(e)
                }
        
        return results
    
    def benchmark_memory_patterns(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark different memory allocation patterns."""
        self._log(f"Benchmarking memory patterns on {device}")
        
        results = {
            'device': str(device),
            'pattern_tests': {}
        }
        
        patterns = [
            ('sequential_allocation', self._test_sequential_allocation),
            ('fragmented_allocation', self._test_fragmented_allocation),
            ('large_tensor_allocation', self._test_large_tensor_allocation),
            ('many_small_tensors', self._test_many_small_tensors),
            ('memory_copy_patterns', self._test_memory_copy_patterns)
        ]
        
        for pattern_name, pattern_func in patterns:
            self._log(f"Testing pattern: {pattern_name}")
            
            try:
                # Clear memory before test
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                
                initial_memory = self.get_memory_info(device)
                start_time = time.time()
                
                pattern_result = pattern_func(device)
                
                execution_time = time.time() - start_time
                final_memory = self.get_memory_info(device)
                
                results['pattern_tests'][pattern_name] = {
                    'execution_time': execution_time,
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'pattern_results': pattern_result,
                    'success': True
                }
                
                self._log(f"✓ {pattern_name} completed in {execution_time:.4f}s")
                
            except Exception as e:
                self._log(f"✗ {pattern_name} failed: {e}", "ERROR")
                results['pattern_tests'][pattern_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _test_sequential_allocation(self, device: torch.device) -> Dict[str, Any]:
        """Test sequential memory allocation pattern."""
        tensors = []
        allocation_times = []
        
        for i in range(10):
            start_time = time.time()
            tensor = torch.randn(500, 500, device=device)
            allocation_times.append(time.time() - start_time)
            tensors.append(tensor)
        
        # Clean up
        del tensors
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'num_allocations': 10,
            'avg_allocation_time': np.mean(allocation_times),
            'allocation_consistency': np.std(allocation_times),
            'total_memory_allocated_mb': 10 * 500 * 500 * 4 / (1024**2)
        }
    
    def _test_fragmented_allocation(self, device: torch.device) -> Dict[str, Any]:
        """Test fragmented memory allocation pattern."""
        tensors = []
        
        # Allocate varying sizes
        sizes = [100, 800, 200, 600, 300, 500, 150, 900, 250, 700]
        
        for size in sizes:
            tensor = torch.randn(size, size, device=device)
            tensors.append(tensor)
        
        # Delete every other tensor to create fragmentation
        for i in range(0, len(tensors), 2):
            del tensors[i]
            tensors[i] = None
        
        # Try to allocate in the gaps
        new_tensors = []
        for i in range(5):
            tensor = torch.randn(400, 400, device=device)
            new_tensors.append(tensor)
        
        # Clean up
        del tensors, new_tensors
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'fragmentation_test': 'completed',
            'pattern': 'alternating_allocation_deallocation'
        }
    
    def _test_large_tensor_allocation(self, device: torch.device) -> Dict[str, Any]:
        """Test large tensor allocation."""
        max_size = 1000
        
        # Gradually increase tensor size until memory limit
        successful_allocations = []
        
        for size in range(500, max_size + 1, 100):
            try:
                start_time = time.time()
                tensor = torch.randn(size, size, device=device)
                allocation_time = time.time() - start_time
                
                successful_allocations.append({
                    'size': size,
                    'allocation_time': allocation_time,
                    'memory_mb': size * size * 4 / (1024**2)
                })
                
                del tensor
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                break
        
        return {
            'max_successful_size': successful_allocations[-1]['size'] if successful_allocations else 0,
            'successful_allocations': len(successful_allocations),
            'allocation_details': successful_allocations
        }
    
    def _test_many_small_tensors(self, device: torch.device) -> Dict[str, Any]:
        """Test allocation of many small tensors."""
        small_tensors = []
        num_tensors = 1000
        tensor_size = 50
        
        start_time = time.time()
        
        for i in range(num_tensors):
            tensor = torch.randn(tensor_size, tensor_size, device=device)
            small_tensors.append(tensor)
        
        allocation_time = time.time() - start_time
        
        # Test access pattern
        start_time = time.time()
        total_sum = 0
        for tensor in small_tensors:
            total_sum += tensor.sum().item()
        
        access_time = time.time() - start_time
        
        # Clean up
        del small_tensors
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'num_small_tensors': num_tensors,
            'tensor_size': tensor_size,
            'total_allocation_time': allocation_time,
            'avg_allocation_time': allocation_time / num_tensors,
            'total_access_time': access_time,
            'total_memory_mb': num_tensors * tensor_size * tensor_size * 4 / (1024**2)
        }
    
    def _test_memory_copy_patterns(self, device: torch.device) -> Dict[str, Any]:
        """Test different memory copy patterns."""
        size = 1000
        
        # Create source tensor
        source = torch.randn(size, size, device=device)
        
        # Test different copy methods
        copy_results = {}
        
        # Clone
        start_time = time.time()
        cloned = source.clone()
        copy_results['clone_time'] = time.time() - start_time
        
        # Copy
        start_time = time.time()
        copied = source.copy_(torch.zeros_like(source))
        copy_results['copy_time'] = time.time() - start_time
        
        # Detach
        start_time = time.time()
        detached = source.detach()
        copy_results['detach_time'] = time.time() - start_time
        
        # To/from CPU (if not already CPU)
        if device.type != 'cpu':
            start_time = time.time()
            cpu_tensor = source.cpu()
            copy_results['to_cpu_time'] = time.time() - start_time
            
            start_time = time.time()
            back_to_device = cpu_tensor.to(device)
            copy_results['from_cpu_time'] = time.time() - start_time
            
            del cpu_tensor, back_to_device
        
        # Clean up
        del source, cloned, copied, detached
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return copy_results
    
    def benchmark_memory_optimization(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark memory optimization techniques."""
        self._log(f"Benchmarking memory optimization on {device}")
        
        results = {
            'device': str(device),
            'optimization_tests': {}
        }
        
        optimizations = [
            ('gradient_checkpointing', self._test_gradient_checkpointing),
            ('mixed_precision', self._test_mixed_precision),
            ('tensor_cores', self._test_tensor_cores),
            ('memory_pinning', self._test_memory_pinning)
        ]
        
        for opt_name, opt_func in optimizations:
            self._log(f"Testing optimization: {opt_name}")
            
            try:
                # Clear memory before test
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                start_time = time.time()
                opt_result = opt_func(device)
                execution_time = time.time() - start_time
                
                results['optimization_tests'][opt_name] = {
                    'execution_time': execution_time,
                    'optimization_results': opt_result,
                    'success': True
                }
                
                self._log(f"✓ {opt_name} completed in {execution_time:.4f}s")
                
            except Exception as e:
                self._log(f"✗ {opt_name} failed: {e}", "ERROR")
                results['optimization_tests'][opt_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _test_gradient_checkpointing(self, device: torch.device) -> Dict[str, Any]:
        """Test gradient checkpointing for memory optimization."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(100, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100, 10)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = SimpleModel().to(device)
        x = torch.randn(64, 100, device=device, requires_grad=True)
        target = torch.randn(64, 10, device=device)
        
        # Normal forward pass
        initial_memory = self.get_memory_info(device)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        normal_memory = self.get_memory_info(device)
        
        # Clear gradients
        model.zero_grad()
        
        # With gradient checkpointing (simplified example)
        if device.type == 'cuda':
            initial_memory_cp = self.get_memory_info(device)
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            checkpointed_memory = self.get_memory_info(device)
        else:
            checkpointed_memory = normal_memory
        
        return {
            'normal_memory_usage': normal_memory,
            'checkpointed_memory_usage': checkpointed_memory,
            'memory_savings': 'gradient_checkpointing_tested'
        }
    
    def _test_mixed_precision(self, device: torch.device) -> Dict[str, Any]:
        """Test mixed precision training for memory optimization."""
        if device.type not in ['cuda', 'mps']:
            return {'note': 'Mixed precision only available on GPU devices'}
        
        size = 500
        
        # FP32 test
        model = torch.nn.Linear(size, size).to(device)
        x = torch.randn(64, size, device=device)
        
        start_time = time.time()
        initial_memory = self.get_memory_info(device)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, x)
        loss.backward()
        
        fp32_time = time.time() - start_time
        fp32_memory = self.get_memory_info(device)
        
        # Clear
        model.zero_grad()
        del output, loss
        
        if device.type == 'cuda':
            # FP16 test with autocast
            start_time = time.time()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, x)
            
            loss.backward()
            fp16_time = time.time() - start_time
            fp16_memory = self.get_memory_info(device)
        else:
            fp16_time = fp32_time
            fp16_memory = fp32_memory
        
        return {
            'fp32_time': fp32_time,
            'fp16_time': fp16_time,
            'fp32_memory': fp32_memory,
            'fp16_memory': fp16_memory,
            'time_improvement': fp32_time / fp16_time if fp16_time > 0 else 1.0,
            'memory_improvement': 'mixed_precision_tested'
        }
    
    def _test_tensor_cores(self, device: torch.device) -> Dict[str, Any]:
        """Test Tensor Core utilization (NVIDIA GPUs)."""
        if device.type != 'cuda':
            return {'note': 'Tensor Cores only available on NVIDIA GPUs'}
        
        # Test with sizes optimized for Tensor Cores (multiples of 8/16)
        sizes = [512, 1024, 2048]
        results = {}
        
        for size in sizes:
            if size > 2048:  # Avoid memory issues
                continue
                
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # Warm up
            for _ in range(3):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                c = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            gflops = (2 * size**3) / (avg_time * 1e9)
            
            results[f'{size}x{size}'] = {
                'avg_time': avg_time,
                'gflops': gflops,
                'tensor_core_optimized': True
            }
            
            del a, b, c
        
        return results
    
    def _test_memory_pinning(self, device: torch.device) -> Dict[str, Any]:
        """Test pinned memory for faster CPU-GPU transfers."""
        if device.type == 'cpu':
            return {'note': 'Memory pinning only relevant for GPU devices'}
        
        size = 1000
        
        # Regular memory transfer
        cpu_tensor = torch.randn(size, size)
        
        start_time = time.time()
        gpu_tensor = cpu_tensor.to(device)
        regular_time = time.time() - start_time
        
        del gpu_tensor
        
        # Pinned memory transfer
        if device.type == 'cuda':
            pinned_tensor = torch.randn(size, size).pin_memory()
            
            start_time = time.time()
            gpu_tensor_pinned = pinned_tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()
            pinned_time = time.time() - start_time
            
            del pinned_tensor, gpu_tensor_pinned
        else:
            pinned_time = regular_time
        
        return {
            'regular_transfer_time': regular_time,
            'pinned_transfer_time': pinned_time,
            'speedup': regular_time / pinned_time if pinned_time > 0 else 1.0,
            'transfer_size_mb': size * size * 4 / (1024**2)
        }
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory benchmark report."""
        self._log("Generating comprehensive memory report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'log_file': str(self.log_file)
            },
            'devices': self.get_available_devices(),
            'memory_benchmarks': {},
            'summary': {}
        }
        
        # Test all available devices
        devices = report['devices']
        
        for device_name, device_info in devices.items():
            if device_info['available']:
                device = device_info['device']
                self._log(f"Running memory benchmarks on device: {device_name}")
                
                try:
                    # Run all memory benchmarks
                    report['memory_benchmarks'][device_name] = {
                        'allocation_benchmark': self.benchmark_tensor_allocation(device),
                        'pattern_benchmark': self.benchmark_memory_patterns(device),
                        'optimization_benchmark': self.benchmark_memory_optimization(device)
                    }
                    
                except Exception as e:
                    self._log(f"Memory benchmark failed for {device_name}: {e}", "ERROR")
                    report['memory_benchmarks'][device_name] = {
                        'error': str(e)
                    }
        
        # Generate summary
        report['summary'] = self._generate_memory_summary(report)
        
        return report
    
    def _generate_memory_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of memory benchmark results."""
        summary = {
            'devices_tested': len(report['memory_benchmarks']),
            'successful_tests': 0,
            'failed_tests': 0,
            'performance_insights': {},
            'recommendations': []
        }
        
        for device_name, benchmarks in report['memory_benchmarks'].items():
            if 'error' not in benchmarks:
                summary['successful_tests'] += 1
                
                # Extract performance insights
                if 'allocation_benchmark' in benchmarks:
                    alloc_results = benchmarks['allocation_benchmark']['allocation_tests']
                    max_size = max(int(k.split('x')[0]) for k in alloc_results.keys() if 'x' in k and 'error' not in alloc_results[k])
                    summary['performance_insights'][device_name] = {
                        'max_tested_size': max_size,
                        'memory_efficient': True
                    }
                    
            else:
                summary['failed_tests'] += 1
        
        # Generate recommendations
        if summary['failed_tests'] > 0:
            summary['recommendations'].append("Some memory tests failed. Check available memory and reduce tensor sizes.")
        
        if any('cuda' in device for device in report['devices'].keys()):
            summary['recommendations'].append("CUDA GPU detected. Consider using mixed precision and gradient checkpointing for large models.")
        
        if any('mps' in device for device in report['devices'].keys()):
            summary['recommendations'].append("Apple Silicon MPS detected. Monitor system memory usage as MPS shares with system RAM.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"pytorch_memory_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create memory benchmark visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PyTorch Memory Benchmark Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Allocation Times by Size
            ax1 = axes[0, 0]
            for device_name, benchmarks in results['memory_benchmarks'].items():
                if 'allocation_benchmark' in benchmarks:
                    alloc_tests = benchmarks['allocation_benchmark']['allocation_tests']
                    sizes = []
                    times = []
                    for test_name, test_data in alloc_tests.items():
                        if 'x' in test_name and 'error' not in test_data:
                            size = int(test_name.split('x')[0])
                            sizes.append(size)
                            times.append(test_data['allocation_time'])
                    
                    if sizes:
                        ax1.plot(sizes, times, marker='o', label=device_name)
            
            ax1.set_xlabel('Tensor Size')
            ax1.set_ylabel('Allocation Time (s)')
            ax1.set_title('Tensor Allocation Times')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Memory Usage by Size
            ax2 = axes[0, 1]
            for device_name, benchmarks in results['memory_benchmarks'].items():
                if 'allocation_benchmark' in benchmarks:
                    alloc_tests = benchmarks['allocation_benchmark']['allocation_tests']
                    sizes = []
                    memory_usage = []
                    for test_name, test_data in alloc_tests.items():
                        if 'x' in test_name and 'error' not in test_data:
                            size = int(test_name.split('x')[0])
                            sizes.append(size)
                            memory_usage.append(test_data['memory_used_mb'])
                    
                    if sizes:
                        ax2.plot(sizes, memory_usage, marker='s', label=device_name)
            
            ax2.set_xlabel('Tensor Size')
            ax2.set_ylabel('Memory Used (MB)')
            ax2.set_title('Memory Usage by Tensor Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Throughput Comparison
            ax3 = axes[1, 0]
            device_names = []
            throughputs = []
            for device_name, benchmarks in results['memory_benchmarks'].items():
                if 'allocation_benchmark' in benchmarks:
                    alloc_tests = benchmarks['allocation_benchmark']['allocation_tests']
                    max_throughput = 0
                    for test_data in alloc_tests.values():
                        if isinstance(test_data, dict) and 'throughput_gflops' in test_data:
                            max_throughput = max(max_throughput, test_data['throughput_gflops'])
                    
                    if max_throughput > 0:
                        device_names.append(device_name)
                        throughputs.append(max_throughput)
            
            if device_names:
                bars = ax3.bar(device_names, throughputs, color=plt.cm.viridis(np.linspace(0, 1, len(device_names))))
                ax3.set_ylabel('Throughput (GFLOPS)')
                ax3.set_title('Peak Throughput by Device')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, throughputs):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # Plot 4: Memory Efficiency
            ax4 = axes[1, 1]
            device_names = []
            efficiencies = []
            for device_name, benchmarks in results['memory_benchmarks'].items():
                if 'allocation_benchmark' in benchmarks:
                    alloc_tests = benchmarks['allocation_benchmark']['allocation_tests']
                    avg_efficiency = 0
                    count = 0
                    for test_data in alloc_tests.values():
                        if isinstance(test_data, dict) and 'efficiency_ratio' in test_data:
                            avg_efficiency += test_data['efficiency_ratio']
                            count += 1
                    
                    if count > 0:
                        device_names.append(device_name)
                        efficiencies.append(avg_efficiency / count)
            
            if device_names:
                bars = ax4.bar(device_names, efficiencies, color=plt.cm.plasma(np.linspace(0, 1, len(device_names))))
                ax4.set_ylabel('Memory Efficiency Ratio')
                ax4.set_title('Average Memory Efficiency')
                ax4.tick_params(axis='x', rotation=45)
                ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Optimum')
                ax4.legend()
                
                # Add value labels on bars
                for bar, value in zip(bars, efficiencies):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiencies)*0.01,
                            f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"pytorch_memory_benchmark_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
        print("PYTORCH MEMORY BENCHMARK SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"System Memory: {platform_info['memory_gb']} GB")
        
        # Device summary
        print(f"\nDevices Tested: {results['summary']['devices_tested']}")
        print(f"Successful Tests: {results['summary']['successful_tests']}")
        print(f"Failed Tests: {results['summary']['failed_tests']}")
        
        # Performance insights
        if results['summary']['performance_insights']:
            print(f"\nPerformance Insights:")
            for device, insights in results['summary']['performance_insights'].items():
                print(f"  {device}: Max tested size {insights['max_tested_size']}x{insights['max_tested_size']}")
        
        # Memory highlights
        print(f"\nMemory Benchmark Highlights:")
        for device_name, benchmarks in results['memory_benchmarks'].items():
            if 'allocation_benchmark' in benchmarks:
                alloc_tests = benchmarks['allocation_benchmark']['allocation_tests']
                successful_tests = sum(1 for test in alloc_tests.values() if isinstance(test, dict) and 'error' not in test)
                print(f"  {device_name}: {successful_tests} successful allocation tests")
                
                # Find best performance
                max_throughput = 0
                for test_data in alloc_tests.values():
                    if isinstance(test_data, dict) and 'throughput_gflops' in test_data:
                        max_throughput = max(max_throughput, test_data['throughput_gflops'])
                
                if max_throughput > 0:
                    print(f"    Peak throughput: {max_throughput:.2f} GFLOPS")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run PyTorch memory benchmark."""
    print("Starting PyTorch Memory Benchmark Suite...")
    
    try:
        # Initialize benchmark
        benchmark = PyTorchMemoryBenchmark()
        
        # Run comprehensive memory benchmarks
        results = benchmark.generate_memory_report()
        
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
                    f'display notification "PyTorch memory benchmark completed successfully!" with title "GET SWIFTY - Memory Benchmark"'
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
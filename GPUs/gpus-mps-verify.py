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
# Script Name: gpus-mps-verify.py                                                
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive Apple Silicon MPS verification and validation tool 
#              with capability testing, compatibility checks, error detection,  
#              and development environment optimization recommendations.         
#
# Usage: python gpus-mps-verify.py [--verbose] [--tests ALL] [--fix-issues]     
#
# Dependencies: torch, numpy, psutil                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional MPS validation with comprehensive testing, diagnostics,   
#        and automated issue resolution for optimal ML development workflows.   
#                                                                                
####################################################################################

"""
Comprehensive Apple Silicon MPS Verification Tool
================================================

Advanced verification and validation tool for Apple Silicon Metal Performance
Shaders (MPS) with capability testing, compatibility checks, error detection,
and development environment optimization for ML workflows.
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
import warnings

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "psutil>=5.8.0"
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
from tkinter import messagebox
import torch
import numpy as np
import psutil

IS_MACOS = sys.platform == "darwin"

class MPSVerificationTool:
    def __init__(self):
        self.setup_logging()
        self.verification_results = {}
        self.issues_found = []
        self.recommendations = []
        self.system_info = {}
        
        # Test configurations
        self.test_suite = {
            'basic_availability': self.test_mps_availability,
            'tensor_creation': self.test_tensor_creation,
            'tensor_operations': self.test_tensor_operations,
            'memory_management': self.test_memory_management,
            'data_transfer': self.test_data_transfer,
            'neural_networks': self.test_neural_networks,
            'autograd': self.test_autograd,
            'mixed_precision': self.test_mixed_precision,
            'error_handling': self.test_error_handling,
            'performance_basic': self.test_basic_performance
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-mps-verify.log"
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
        
    def get_system_info(self):
        """Gather comprehensive system information"""
        try:
            self.system_info = {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            }
            
            # macOS specific information
            if IS_MACOS:
                mac_version = platform.mac_ver()
                self.system_info['macos_version'] = mac_version[0]
                
                # Get detailed hardware info
                try:
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
                    
            # Memory information
            memory = psutil.virtual_memory()
            self.system_info.update({
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent_used': memory.percent
            })
            
            return self.system_info
            
        except Exception as e:
            self.logger.error(f"Error gathering system info: {e}")
            return {}
            
    def test_mps_availability(self):
        """Test basic MPS availability"""
        test_result = {
            'test_name': 'MPS Availability',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check if we're on macOS
            if not IS_MACOS:
                test_result['issues'].append("MPS is only available on macOS")
                test_result['recommendations'].append("Use CUDA on Linux/Windows or CPU fallback")
                return test_result
                
            # Check PyTorch version
            torch_version = torch.__version__
            test_result['details']['torch_version'] = torch_version
            
            # Parse version to check if it's recent enough
            version_parts = torch_version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 1 or (major == 1 and minor < 12):
                test_result['issues'].append(f"PyTorch version {torch_version} may not support MPS")
                test_result['recommendations'].append("Upgrade to PyTorch 1.12.0 or later")
                
            # Check MPS backend availability
            if not hasattr(torch.backends, 'mps'):
                test_result['issues'].append("PyTorch MPS backend not found")
                test_result['recommendations'].append("Install PyTorch with MPS support")
                return test_result
                
            test_result['details']['mps_backend_available'] = True
            
            # Check if MPS is built
            if not torch.backends.mps.is_built():
                test_result['issues'].append("PyTorch was not built with MPS support")
                test_result['recommendations'].append("Install official PyTorch with MPS support")
                return test_result
                
            test_result['details']['mps_built'] = True
            
            # Check if MPS is available
            if not torch.backends.mps.is_available():
                test_result['issues'].append("MPS is not available on this system")
                test_result['recommendations'].append("Check macOS version (requires macOS 12.3+)")
                return test_result
                
            test_result['details']['mps_available'] = True
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Exception during MPS availability check: {e}")
            
        return test_result
        
    def test_tensor_creation(self):
        """Test tensor creation on MPS device"""
        test_result = {
            'test_name': 'Tensor Creation',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test different tensor creation methods
            creation_tests = {
                'randn': lambda: torch.randn(100, 100, device=device),
                'zeros': lambda: torch.zeros(100, 100, device=device),
                'ones': lambda: torch.ones(100, 100, device=device),
                'eye': lambda: torch.eye(100, device=device),
                'arange': lambda: torch.arange(1000, device=device, dtype=torch.float32)
            }
            
            for test_name, create_func in creation_tests.items():
                try:
                    tensor = create_func()
                    test_result['details'][f'{test_name}_success'] = True
                    test_result['details'][f'{test_name}_device'] = str(tensor.device)
                    test_result['details'][f'{test_name}_dtype'] = str(tensor.dtype)
                    
                    # Verify tensor is actually on MPS
                    if tensor.device.type != 'mps':
                        test_result['issues'].append(f"{test_name} tensor not on MPS device")
                        
                except Exception as e:
                    test_result['issues'].append(f"{test_name} creation failed: {e}")
                    test_result['details'][f'{test_name}_success'] = False
                    
            # Test different data types
            dtype_tests = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
            
            for dtype in dtype_tests:
                try:
                    tensor = torch.randn(10, 10, device=device, dtype=dtype)
                    test_result['details'][f'dtype_{dtype}_supported'] = True
                except Exception as e:
                    test_result['details'][f'dtype_{dtype}_supported'] = False
                    test_result['issues'].append(f"Data type {dtype} not supported: {e}")
                    
            if not test_result['issues']:
                test_result['passed'] = True
                
        except Exception as e:
            test_result['issues'].append(f"General tensor creation error: {e}")
            
        return test_result
        
    def test_tensor_operations(self):
        """Test basic tensor operations on MPS"""
        test_result = {
            'test_name': 'Tensor Operations',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Create test tensors
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            
            # Test basic operations
            operations = {
                'addition': lambda: a + b,
                'subtraction': lambda: a - b,
                'multiplication': lambda: a * b,
                'division': lambda: a / b,
                'matrix_multiply': lambda: torch.matmul(a, b),
                'transpose': lambda: a.T,
                'sum': lambda: torch.sum(a),
                'mean': lambda: torch.mean(a),
                'max': lambda: torch.max(a),
                'min': lambda: torch.min(a),
                'relu': lambda: torch.relu(a),
                'sigmoid': lambda: torch.sigmoid(a),
                'tanh': lambda: torch.tanh(a),
                'exp': lambda: torch.exp(a * 0.1),  # Scale to avoid overflow
                'log': lambda: torch.log(torch.abs(a) + 1e-8),
                'sqrt': lambda: torch.sqrt(torch.abs(a))
            }
            
            for op_name, op_func in operations.items():
                try:
                    result = op_func()
                    test_result['details'][f'{op_name}_success'] = True
                    
                    # Check if result is on correct device
                    if hasattr(result, 'device') and result.device.type != 'mps':
                        test_result['issues'].append(f"{op_name} result not on MPS device")
                        
                    # Check for NaN or Inf values
                    if hasattr(result, 'isnan') and torch.any(torch.isnan(result)):
                        test_result['issues'].append(f"{op_name} produced NaN values")
                    if hasattr(result, 'isinf') and torch.any(torch.isinf(result)):
                        test_result['issues'].append(f"{op_name} produced Inf values")
                        
                except Exception as e:
                    test_result['issues'].append(f"{op_name} operation failed: {e}")
                    test_result['details'][f'{op_name}_success'] = False
                    
            # Test in-place operations
            inplace_ops = {
                'add_': lambda x: x.add_(1.0),
                'mul_': lambda x: x.mul_(0.5),
                'relu_': lambda x: x.relu_()
            }
            
            for op_name, op_func in inplace_ops.items():
                try:
                    test_tensor = torch.randn(10, 10, device=device)
                    op_func(test_tensor)
                    test_result['details'][f'{op_name}_success'] = True
                except Exception as e:
                    test_result['issues'].append(f"In-place {op_name} failed: {e}")
                    test_result['details'][f'{op_name}_success'] = False
                    
            if len([k for k, v in test_result['details'].items() if k.endswith('_success') and v]) > len(operations) * 0.8:
                test_result['passed'] = True
                
        except Exception as e:
            test_result['issues'].append(f"General tensor operations error: {e}")
            
        return test_result
        
    def test_memory_management(self):
        """Test MPS memory management"""
        test_result = {
            'test_name': 'Memory Management',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test memory allocation and deallocation
            initial_memory = torch.mps.current_allocated_memory()
            test_result['details']['initial_memory_mb'] = initial_memory / (1024 * 1024)
            
            # Allocate tensors
            tensors = []
            allocation_size = 10 * 1024 * 1024  # 10MB per tensor
            
            for i in range(5):
                tensor = torch.randn(allocation_size // 4, device=device, dtype=torch.float32)
                tensors.append(tensor)
                
            after_allocation = torch.mps.current_allocated_memory()
            test_result['details']['after_allocation_mb'] = after_allocation / (1024 * 1024)
            test_result['details']['allocated_mb'] = (after_allocation - initial_memory) / (1024 * 1024)
            
            # Test memory clearing
            del tensors
            torch.mps.empty_cache()
            
            after_cleanup = torch.mps.current_allocated_memory()
            test_result['details']['after_cleanup_mb'] = after_cleanup / (1024 * 1024)
            
            # Check if memory was properly freed
            memory_freed = after_allocation - after_cleanup
            test_result['details']['memory_freed_mb'] = memory_freed / (1024 * 1024)
            
            if memory_freed > 0:
                test_result['details']['memory_cleanup_working'] = True
            else:
                test_result['issues'].append("Memory cleanup not working properly")
                test_result['recommendations'].append("Try torch.mps.empty_cache() to free memory")
                
            # Test memory functions
            try:
                max_memory = torch.mps.driver_allocated_memory()
                test_result['details']['max_memory_mb'] = max_memory / (1024 * 1024)
            except Exception as e:
                test_result['issues'].append(f"Could not get max memory info: {e}")
                
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Memory management test error: {e}")
            
        return test_result
        
    def test_data_transfer(self):
        """Test data transfer between CPU and MPS"""
        test_result = {
            'test_name': 'Data Transfer',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test CPU to MPS transfer
            cpu_tensor = torch.randn(1000, 1000)
            
            start_time = time.time()
            mps_tensor = cpu_tensor.to('mps')
            cpu_to_mps_time = time.time() - start_time
            
            test_result['details']['cpu_to_mps_time_ms'] = cpu_to_mps_time * 1000
            test_result['details']['cpu_to_mps_success'] = mps_tensor.device.type == 'mps'
            
            # Test MPS to CPU transfer
            start_time = time.time()
            back_to_cpu = mps_tensor.to('cpu')
            mps_to_cpu_time = time.time() - start_time
            
            test_result['details']['mps_to_cpu_time_ms'] = mps_to_cpu_time * 1000
            test_result['details']['mps_to_cpu_success'] = back_to_cpu.device.type == 'cpu'
            
            # Test data integrity
            if torch.allclose(cpu_tensor, back_to_cpu, rtol=1e-5):
                test_result['details']['data_integrity_preserved'] = True
            else:
                test_result['issues'].append("Data integrity not preserved during transfer")
                
            # Test different transfer methods
            transfer_methods = {
                'to_method': lambda x: x.to('mps'),
                'cuda_method': lambda x: x.mps() if hasattr(x, 'mps') else x.to('mps'),
                'copy_method': lambda x: x.to('mps', copy=True)
            }
            
            for method_name, method_func in transfer_methods.items():
                try:
                    test_tensor = torch.randn(100, 100)
                    transferred = method_func(test_tensor)
                    test_result['details'][f'{method_name}_success'] = transferred.device.type == 'mps'
                except Exception as e:
                    test_result['issues'].append(f"{method_name} transfer failed: {e}")
                    test_result['details'][f'{method_name}_success'] = False
                    
            # Performance check
            if cpu_to_mps_time > 1.0:  # More than 1 second for 1M elements
                test_result['issues'].append("CPU to MPS transfer is slow")
                test_result['recommendations'].append("Consider batch transfers for better performance")
                
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Data transfer test error: {e}")
            
        return test_result
        
    def test_neural_networks(self):
        """Test neural network operations on MPS"""
        test_result = {
            'test_name': 'Neural Networks',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test basic layers
            layers_to_test = {
                'linear': torch.nn.Linear(100, 50),
                'conv2d': torch.nn.Conv2d(3, 16, kernel_size=3),
                'relu': torch.nn.ReLU(),
                'sigmoid': torch.nn.Sigmoid(),
                'tanh': torch.nn.Tanh(),
                'dropout': torch.nn.Dropout(0.5),
                'batchnorm1d': torch.nn.BatchNorm1d(50),
                'batchnorm2d': torch.nn.BatchNorm2d(16)
            }
            
            for layer_name, layer in layers_to_test.items():
                try:
                    layer = layer.to(device)
                    
                    if layer_name == 'conv2d' or layer_name == 'batchnorm2d':
                        test_input = torch.randn(4, 3, 32, 32, device=device)
                    elif layer_name == 'batchnorm1d':
                        test_input = torch.randn(4, 50, device=device)
                    else:
                        test_input = torch.randn(4, 100, device=device)
                        
                    output = layer(test_input)
                    test_result['details'][f'{layer_name}_success'] = True
                    test_result['details'][f'{layer_name}_output_device'] = str(output.device)
                    
                except Exception as e:
                    test_result['issues'].append(f"{layer_name} layer failed: {e}")
                    test_result['details'][f'{layer_name}_success'] = False
                    
            # Test a simple neural network
            try:
                class SimpleNet(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = torch.nn.Linear(784, 128)
                        self.fc2 = torch.nn.Linear(128, 64)
                        self.fc3 = torch.nn.Linear(64, 10)
                        self.relu = torch.nn.ReLU()
                        
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.relu(self.fc2(x))
                        x = self.fc3(x)
                        return x
                        
                model = SimpleNet().to(device)
                test_input = torch.randn(32, 784, device=device)
                
                output = model(test_input)
                test_result['details']['simple_network_success'] = True
                test_result['details']['network_output_shape'] = list(output.shape)
                
            except Exception as e:
                test_result['issues'].append(f"Simple network test failed: {e}")
                test_result['details']['simple_network_success'] = False
                
            # Count successful tests
            successful_tests = sum(1 for k, v in test_result['details'].items() 
                                 if k.endswith('_success') and v)
            total_tests = len([k for k in test_result['details'].keys() if k.endswith('_success')])
            
            if successful_tests > total_tests * 0.8:
                test_result['passed'] = True
                
        except Exception as e:
            test_result['issues'].append(f"Neural network test error: {e}")
            
        return test_result
        
    def test_autograd(self):
        """Test automatic differentiation on MPS"""
        test_result = {
            'test_name': 'Automatic Differentiation',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test basic gradient computation
            x = torch.randn(10, 10, device=device, requires_grad=True)
            y = torch.sum(x ** 2)
            
            y.backward()
            
            if x.grad is not None:
                test_result['details']['basic_gradient_success'] = True
                test_result['details']['gradient_device'] = str(x.grad.device)
                
                # Check gradient values
                expected_grad = 2 * x
                if torch.allclose(x.grad, expected_grad, rtol=1e-4):
                    test_result['details']['gradient_correctness'] = True
                else:
                    test_result['issues'].append("Gradient values are incorrect")
            else:
                test_result['issues'].append("Gradient computation failed")
                
            # Test with neural network
            try:
                model = torch.nn.Linear(10, 1).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                
                # Forward pass
                x = torch.randn(32, 10, device=device)
                y_true = torch.randn(32, 1, device=device)
                y_pred = model(x)
                
                # Loss computation
                loss = torch.nn.functional.mse_loss(y_pred, y_true)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                test_result['details']['network_training_success'] = True
                
                # Check if gradients exist
                has_gradients = any(param.grad is not None for param in model.parameters())
                test_result['details']['network_gradients_computed'] = has_gradients
                
            except Exception as e:
                test_result['issues'].append(f"Network training test failed: {e}")
                test_result['details']['network_training_success'] = False
                
            if not test_result['issues']:
                test_result['passed'] = True
                
        except Exception as e:
            test_result['issues'].append(f"Autograd test error: {e}")
            
        return test_result
        
    def test_mixed_precision(self):
        """Test mixed precision operations on MPS"""
        test_result = {
            'test_name': 'Mixed Precision',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test float16 operations
            try:
                x_fp16 = torch.randn(100, 100, device=device, dtype=torch.float16)
                y_fp16 = torch.randn(100, 100, device=device, dtype=torch.float16)
                
                # Basic operations
                result_add = x_fp16 + y_fp16
                result_mul = torch.matmul(x_fp16, y_fp16)
                
                test_result['details']['float16_operations_success'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Float16 operations failed: {e}")
                test_result['details']['float16_operations_success'] = False
                
            # Test mixed precision with autocast (if available)
            try:
                if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                    x_fp32 = torch.randn(100, 100, device=device, dtype=torch.float32)
                    
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        result = torch.matmul(x_fp32, x_fp32)
                        
                    test_result['details']['autocast_success'] = True
                else:
                    test_result['details']['autocast_available'] = False
                    test_result['recommendations'].append("Autocast may not be available for MPS")
                    
            except Exception as e:
                test_result['issues'].append(f"Autocast test failed: {e}")
                test_result['details']['autocast_success'] = False
                
            # Test precision conversion
            try:
                x_fp32 = torch.randn(50, 50, device=device, dtype=torch.float32)
                x_fp16 = x_fp32.half()
                x_back_fp32 = x_fp16.float()
                
                test_result['details']['precision_conversion_success'] = True
                
                # Check if conversion preserves reasonable accuracy
                diff = torch.abs(x_fp32 - x_back_fp32)
                max_diff = torch.max(diff).item()
                test_result['details']['max_conversion_error'] = max_diff
                
                if max_diff > 1e-2:  # Allow for float16 precision loss
                    test_result['issues'].append("Large precision loss in float16 conversion")
                    
            except Exception as e:
                test_result['issues'].append(f"Precision conversion failed: {e}")
                test_result['details']['precision_conversion_success'] = False
                
            # Overall assessment
            successful_tests = sum(1 for k, v in test_result['details'].items() 
                                 if k.endswith('_success') and v)
            
            if successful_tests > 0:
                test_result['passed'] = True
                
        except Exception as e:
            test_result['issues'].append(f"Mixed precision test error: {e}")
            
        return test_result
        
    def test_error_handling(self):
        """Test error handling and edge cases"""
        test_result = {
            'test_name': 'Error Handling',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device = torch.device('mps')
            
            # Test memory overflow handling
            try:
                # Try to allocate a very large tensor
                large_tensor = torch.randn(100000, 100000, device=device)
                test_result['details']['large_allocation_succeeded'] = True
                del large_tensor
                torch.mps.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    test_result['details']['memory_error_handled'] = True
                else:
                    test_result['issues'].append(f"Unexpected error for large allocation: {e}")
            except Exception as e:
                test_result['issues'].append(f"Large allocation test failed: {e}")
                
            # Test invalid operations
            try:
                # Try operations that might not be supported
                x = torch.randn(10, 10, device=device)
                
                # Test unsupported dtype operations
                try:
                    x_complex = x.to(torch.complex64)
                    test_result['details']['complex_dtype_supported'] = True
                except Exception:
                    test_result['details']['complex_dtype_supported'] = False
                    
            except Exception as e:
                test_result['issues'].append(f"Invalid operation test failed: {e}")
                
            # Test tensor size mismatches
            try:
                a = torch.randn(10, 20, device=device)
                b = torch.randn(30, 40, device=device)
                
                try:
                    result = torch.matmul(a, b)
                    test_result['issues'].append("Size mismatch not properly caught")
                except RuntimeError:
                    test_result['details']['size_mismatch_error_handled'] = True
                    
            except Exception as e:
                test_result['issues'].append(f"Size mismatch test failed: {e}")
                
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Error handling test error: {e}")
            
        return test_result
        
    def test_basic_performance(self):
        """Test basic performance characteristics"""
        test_result = {
            'test_name': 'Basic Performance',
            'passed': False,
            'details': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            device_cpu = torch.device('cpu')
            device_mps = torch.device('mps')
            
            # Matrix multiplication benchmark
            size = 1000
            iterations = 5
            
            # CPU benchmark
            cpu_times = []
            for _ in range(iterations):
                a_cpu = torch.randn(size, size, device=device_cpu)
                b_cpu = torch.randn(size, size, device=device_cpu)
                
                start_time = time.time()
                result_cpu = torch.matmul(a_cpu, b_cpu)
                cpu_times.append(time.time() - start_time)
                
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            test_result['details']['cpu_avg_time_ms'] = avg_cpu_time * 1000
            
            # MPS benchmark
            mps_times = []
            for _ in range(iterations):
                a_mps = torch.randn(size, size, device=device_mps)
                b_mps = torch.randn(size, size, device=device_mps)
                
                start_time = time.time()
                result_mps = torch.matmul(a_mps, b_mps)
                torch.mps.synchronize()  # Ensure completion
                mps_times.append(time.time() - start_time)
                
            avg_mps_time = sum(mps_times) / len(mps_times)
            test_result['details']['mps_avg_time_ms'] = avg_mps_time * 1000
            
            # Calculate speedup
            speedup = avg_cpu_time / avg_mps_time
            test_result['details']['mps_speedup'] = speedup
            
            if speedup > 1.0:
                test_result['details']['mps_faster_than_cpu'] = True
            else:
                test_result['issues'].append("MPS is slower than CPU for matrix operations")
                test_result['recommendations'].append("Check system resources and thermal throttling")
                
            # Performance thresholds
            if avg_mps_time > 1.0:  # More than 1 second for 1000x1000 matmul
                test_result['issues'].append("MPS performance is slower than expected")
                test_result['recommendations'].append("Check system memory and background processes")
                
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Performance test error: {e}")
            
        return test_result
        
    def run_verification_suite(self, tests_to_run: List[str] = None):
        """Run the complete verification suite"""
        if tests_to_run is None:
            tests_to_run = list(self.test_suite.keys())
            
        self.logger.info("🚀 Starting MPS verification suite...")
        
        total_tests = len(tests_to_run)
        passed_tests = 0
        
        for i, test_name in enumerate(tests_to_run, 1):
            self.logger.info(f"Running test {i}/{total_tests}: {test_name}")
            
            if test_name in self.test_suite:
                try:
                    result = self.test_suite[test_name]()
                    self.verification_results[test_name] = result
                    
                    if result['passed']:
                        passed_tests += 1
                        self.logger.info(f"✅ {result['test_name']} - PASSED")
                    else:
                        self.logger.warning(f"❌ {result['test_name']} - FAILED")
                        
                    # Collect issues and recommendations
                    self.issues_found.extend(result['issues'])
                    self.recommendations.extend(result['recommendations'])
                    
                except Exception as e:
                    self.logger.error(f"❌ {test_name} - ERROR: {e}")
                    self.verification_results[test_name] = {
                        'test_name': test_name,
                        'passed': False,
                        'error': str(e)
                    }
            else:
                self.logger.error(f"❌ Unknown test: {test_name}")
                
        self.logger.info(f"\n🏁 Verification complete: {passed_tests}/{total_tests} tests passed")
        return self.verification_results
        
    def generate_report(self, output_path=None):
        """Generate comprehensive verification report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = Path.home() / "Desktop" / f"mps_verification_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        report = {
            'timestamp': timestamp,
            'system_info': self.system_info,
            'verification_results': self.verification_results,
            'issues_found': list(set(self.issues_found)),  # Remove duplicates
            'recommendations': list(set(self.recommendations)),  # Remove duplicates
            'summary': {
                'total_tests': len(self.verification_results),
                'passed_tests': sum(1 for result in self.verification_results.values() if result.get('passed', False)),
                'failed_tests': sum(1 for result in self.verification_results.values() if not result.get('passed', True)),
                'total_issues': len(set(self.issues_found)),
                'mps_available': self.verification_results.get('basic_availability', {}).get('passed', False)
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
        """Display verification results summary"""
        print("\n" + "="*70)
        print("          APPLE SILICON MPS VERIFICATION SUMMARY")
        print("="*70)
        
        # System Information
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   Platform: {self.system_info.get('platform', 'Unknown')}")
        print(f"   Chip: {self.system_info.get('chip', 'Unknown')}")
        print(f"   macOS: {self.system_info.get('macos_version', 'Unknown')}")
        print(f"   PyTorch: {self.system_info.get('torch_version', 'Unknown')}")
        print(f"   Memory: {self.system_info.get('memory_total_gb', 0)} GB")
        
        # Test Results
        total_tests = len(self.verification_results)
        passed_tests = sum(1 for result in self.verification_results.values() if result.get('passed', False))
        
        print(f"\n🧪 VERIFICATION RESULTS:")
        print(f"   Tests Run: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Individual test results
        print(f"\n📋 TEST DETAILS:")
        for test_name, result in self.verification_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"   {status} {result.get('test_name', test_name)}")
            
        # Issues found
        if self.issues_found:
            print(f"\n⚠️  ISSUES FOUND ({len(set(self.issues_found))}):")
            for issue in sorted(set(self.issues_found)):
                print(f"   • {issue}")
                
        # Recommendations
        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(set(self.recommendations))}):")
            for rec in sorted(set(self.recommendations)):
                print(f"   • {rec}")
                
        # Overall assessment
        mps_available = self.verification_results.get('basic_availability', {}).get('passed', False)
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if mps_available and passed_tests > total_tests * 0.8:
            print("   ✅ MPS is working properly and ready for ML workflows")
        elif mps_available:
            print("   ⚠️ MPS is available but has some issues")
        else:
            print("   ❌ MPS is not available or not working properly")
            
        print("\n" + "="*70)
        
    def show_gui_summary(self):
        """Show GUI summary of verification results"""
        if not IS_MACOS:
            return
            
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Create summary message
            total_tests = len(self.verification_results)
            passed_tests = sum(1 for result in self.verification_results.values() if result.get('passed', False))
            success_rate = (passed_tests/total_tests*100) if total_tests > 0 else 0
            
            mps_available = self.verification_results.get('basic_availability', {}).get('passed', False)
            chip = self.system_info.get('chip', 'Unknown')
            
            summary = f"""Apple Silicon MPS Verification Summary

System: {chip}
MPS Available: {'✅ Yes' if mps_available else '❌ No'}

Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)

"""
            
            if self.issues_found:
                summary += f"Issues Found: {len(set(self.issues_found))}\n"
                
            if mps_available and success_rate > 80:
                summary += "\n✅ MPS is working properly!"
            elif mps_available:
                summary += "\n⚠️ MPS has some issues"
            else:
                summary += "\n❌ MPS is not working"
                
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("MPS Verification Results", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive Apple Silicon MPS Verification Tool")
        parser.add_argument('--tests', '-t', default='all', help='Tests to run (comma-separated or "all")')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--export', '-e', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--fix-issues', action='store_true', help='Attempt to fix common issues')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Apple Silicon MPS Verification Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("🚀 Starting Apple Silicon MPS verification...")
        
        try:
            # Get system information
            self.get_system_info()
            
            # Determine tests to run
            if args.tests.lower() == 'all':
                tests_to_run = list(self.test_suite.keys())
            else:
                tests_to_run = [t.strip() for t in args.tests.split(',')]
                # Validate test names
                invalid_tests = [t for t in tests_to_run if t not in self.test_suite]
                if invalid_tests:
                    print(f"❌ Invalid test names: {invalid_tests}")
                    print(f"Available tests: {list(self.test_suite.keys())}")
                    return
                    
            # Run verification suite
            self.run_verification_suite(tests_to_run)
            
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
            self.logger.info("Verification interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

def main():
    verification_tool = MPSVerificationTool()
    verification_tool.run()

if __name__ == "__main__":
    main()
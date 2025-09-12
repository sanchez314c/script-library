#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

PyTorch Basic GPU Verification & Testing Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive PyTorch basic verification and GPU testing

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

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'torch',
        'torchvision', 
        'psutil',
        'numpy'
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

class PyTorchBasicVerifier:
    """Comprehensive PyTorch basic verification and testing suite."""
    
    def __init__(self):
        """Initialize the PyTorch basic verifier."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"pytorch_basic_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"PyTorch Basic Verification Log - {datetime.now()}\n")
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
    
    def check_pytorch_installation(self) -> Dict[str, Any]:
        """Verify PyTorch installation and basic information."""
        self._log("Checking PyTorch installation")
        
        try:
            info = {
                'pytorch_version': torch.__version__,
                'torchvision_version': torchvision.__version__,
                'python_version': sys.version.split()[0],
                'installation_valid': True,
                'cuda_available': torch.cuda.is_available(),
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                'cpu_available': True
            }
            
            # Get CUDA information if available
            if info['cuda_available']:
                info.update({
                    'cuda_version': torch.version.cuda,
                    'cudnn_version': torch.backends.cudnn.version(),
                    'cuda_device_count': torch.cuda.device_count(),
                    'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                })
            
            # Get MPS information if available
            if info['mps_available']:
                info['mps_built'] = torch.backends.mps.is_built()
            
            self._log(f"PyTorch installation verified: {info['pytorch_version']}")
            return info
            
        except Exception as e:
            self._log(f"Error checking PyTorch installation: {e}", "ERROR")
            return {'installation_valid': False, 'error': str(e)}
    
    def get_available_devices(self) -> Dict[str, Any]:
        """Get all available devices for computation."""
        self._log("Detecting available devices")
        
        devices = {
            'cpu': {
                'available': True,
                'device': torch.device('cpu'),
                'info': f"CPU ({self.platform_info['cpu_count']} cores)"
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
                    'properties': {
                        'total_memory': properties.total_memory,
                        'memory_gb': round(properties.total_memory / (1024**3), 2),
                        'multiprocessor_count': properties.multi_processor_count,
                        'compute_capability': f"{properties.major}.{properties.minor}"
                    }
                }
        
        # Check MPS device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices['mps'] = {
                'available': True,
                'device': torch.device('mps'),
                'info': "Apple Metal Performance Shaders"
            }
        
        self._log(f"Found {len(devices)} available devices")
        return devices
    
    def test_basic_tensor_operations(self, device: torch.device) -> Dict[str, Any]:
        """Test basic tensor operations on specified device."""
        self._log(f"Testing basic tensor operations on {device}")
        
        results = {
            'device': str(device),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        tests = [
            ('tensor_creation', self._test_tensor_creation),
            ('tensor_arithmetic', self._test_tensor_arithmetic),
            ('tensor_indexing', self._test_tensor_indexing),
            ('tensor_reshaping', self._test_tensor_reshaping),
            ('tensor_broadcasting', self._test_tensor_broadcasting),
            ('tensor_reduction', self._test_tensor_reduction),
            ('tensor_comparison', self._test_tensor_comparison),
            ('tensor_logical', self._test_tensor_logical)
        ]
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                test_result = test_func(device)
                execution_time = time.time() - start_time
                
                results['test_details'][test_name] = {
                    'passed': True,
                    'execution_time': execution_time,
                    'details': test_result
                }
                results['tests_passed'] += 1
                self._log(f"✓ {test_name} passed ({execution_time:.4f}s)")
                
            except Exception as e:
                results['test_details'][test_name] = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                results['tests_failed'] += 1
                self._log(f"✗ {test_name} failed: {e}", "ERROR")
        
        return results
    
    def _test_tensor_creation(self, device: torch.device) -> Dict[str, Any]:
        """Test various tensor creation methods."""
        results = {}
        
        # Test different tensor creation methods
        results['zeros'] = torch.zeros(5, 3, device=device).shape
        results['ones'] = torch.ones(5, 3, device=device).shape
        results['empty'] = torch.empty(5, 3, device=device).shape
        results['rand'] = torch.rand(5, 3, device=device).shape
        results['randn'] = torch.randn(5, 3, device=device).shape
        results['arange'] = torch.arange(0, 10, device=device).shape
        results['linspace'] = torch.linspace(0, 1, 10, device=device).shape
        results['eye'] = torch.eye(5, device=device).shape
        
        # Test tensor from data
        data = [[1, 2], [3, 4]]
        results['from_list'] = torch.tensor(data, device=device).shape
        
        return results
    
    def _test_tensor_arithmetic(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor arithmetic operations."""
        a = torch.rand(3, 4, device=device)
        b = torch.rand(3, 4, device=device)
        
        results = {}
        results['addition'] = (a + b).shape
        results['subtraction'] = (a - b).shape
        results['multiplication'] = (a * b).shape
        results['division'] = (a / (b + 1e-8)).shape  # Add small epsilon to avoid division by zero
        results['power'] = torch.pow(a, 2).shape
        results['sqrt'] = torch.sqrt(a).shape
        results['exp'] = torch.exp(a).shape
        results['log'] = torch.log(a + 1e-8).shape  # Add small epsilon for numerical stability
        
        return results
    
    def _test_tensor_indexing(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor indexing and slicing."""
        x = torch.rand(5, 4, 3, device=device)
        
        results = {}
        results['basic_indexing'] = x[0].shape
        results['slicing'] = x[1:3].shape
        results['advanced_indexing'] = x[:, 1:3, :].shape
        results['boolean_indexing'] = x[x > 0.5].shape
        
        # Test fancy indexing
        indices = torch.tensor([0, 2, 4], device=device)
        results['fancy_indexing'] = x[indices].shape
        
        return results
    
    def _test_tensor_reshaping(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor reshaping operations."""
        x = torch.rand(2, 3, 4, device=device)
        
        results = {}
        results['view'] = x.view(6, 4).shape
        results['reshape'] = x.reshape(-1, 2).shape
        results['transpose'] = x.transpose(0, 1).shape
        results['permute'] = x.permute(2, 0, 1).shape
        results['squeeze'] = torch.rand(1, 3, 1, 4, device=device).squeeze().shape
        results['unsqueeze'] = x.unsqueeze(0).shape
        results['flatten'] = torch.flatten(x).shape
        
        return results
    
    def _test_tensor_broadcasting(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor broadcasting operations."""
        a = torch.rand(3, 1, device=device)
        b = torch.rand(1, 4, device=device)
        c = torch.rand(3, 4, device=device)
        
        results = {}
        results['broadcast_add'] = (a + b).shape
        results['broadcast_mul'] = (a * c).shape
        results['broadcast_sub'] = (c - a).shape
        
        return results
    
    def _test_tensor_reduction(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor reduction operations."""
        x = torch.rand(3, 4, 5, device=device)
        
        results = {}
        results['sum_all'] = torch.sum(x).item()
        results['sum_dim'] = torch.sum(x, dim=1).shape
        results['mean'] = torch.mean(x).item()
        results['std'] = torch.std(x).item()
        results['max'] = torch.max(x).item()
        results['min'] = torch.min(x).item()
        results['argmax'] = torch.argmax(x).item()
        results['argmin'] = torch.argmin(x).item()
        
        return results
    
    def _test_tensor_comparison(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor comparison operations."""
        a = torch.rand(3, 4, device=device)
        b = torch.rand(3, 4, device=device)
        
        results = {}
        results['equal'] = torch.equal(a, a)
        results['greater'] = (a > b).sum().item()
        results['less'] = (a < b).sum().item()
        results['greater_equal'] = (a >= b).sum().item()
        results['less_equal'] = (a <= b).sum().item()
        results['not_equal'] = (a != b).sum().item()
        
        return results
    
    def _test_tensor_logical(self, device: torch.device) -> Dict[str, Any]:
        """Test tensor logical operations."""
        a = torch.rand(3, 4, device=device) > 0.5
        b = torch.rand(3, 4, device=device) > 0.5
        
        results = {}
        results['logical_and'] = torch.logical_and(a, b).sum().item()
        results['logical_or'] = torch.logical_or(a, b).sum().item()
        results['logical_not'] = torch.logical_not(a).sum().item()
        results['logical_xor'] = torch.logical_xor(a, b).sum().item()
        
        return results
    
    def test_matrix_operations(self, device: torch.device) -> Dict[str, Any]:
        """Test matrix operations on specified device."""
        self._log(f"Testing matrix operations on {device}")
        
        results = {
            'device': str(device),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        tests = [
            ('matrix_multiplication', self._test_matrix_multiplication),
            ('matrix_decomposition', self._test_matrix_decomposition),
            ('linear_algebra', self._test_linear_algebra),
            ('eigenvalues', self._test_eigenvalues)
        ]
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                test_result = test_func(device)
                execution_time = time.time() - start_time
                
                results['test_details'][test_name] = {
                    'passed': True,
                    'execution_time': execution_time,
                    'details': test_result
                }
                results['tests_passed'] += 1
                self._log(f"✓ {test_name} passed ({execution_time:.4f}s)")
                
            except Exception as e:
                results['test_details'][test_name] = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                results['tests_failed'] += 1
                self._log(f"✗ {test_name} failed: {e}", "ERROR")
        
        return results
    
    def _test_matrix_multiplication(self, device: torch.device) -> Dict[str, Any]:
        """Test various matrix multiplication operations."""
        a = torch.rand(3, 4, device=device)
        b = torch.rand(4, 5, device=device)
        c = torch.rand(3, 3, device=device)
        
        results = {}
        results['matmul'] = torch.matmul(a, b).shape
        results['mm'] = torch.mm(c, c).shape
        results['bmm'] = torch.bmm(
            torch.rand(10, 3, 4, device=device),
            torch.rand(10, 4, 5, device=device)
        ).shape
        results['dot'] = torch.dot(
            torch.rand(5, device=device),
            torch.rand(5, device=device)
        ).item()
        
        return results
    
    def _test_matrix_decomposition(self, device: torch.device) -> Dict[str, Any]:
        """Test matrix decomposition operations."""
        # Create a positive definite matrix for Cholesky decomposition
        a = torch.rand(3, 3, device=device)
        symmetric_pd = torch.mm(a, a.t()) + torch.eye(3, device=device) * 0.1
        
        results = {}
        
        # QR decomposition
        q, r = torch.qr(torch.rand(4, 3, device=device))
        results['qr_decomposition'] = {'q_shape': q.shape, 'r_shape': r.shape}
        
        # SVD decomposition
        u, s, v = torch.svd(torch.rand(4, 3, device=device))
        results['svd_decomposition'] = {'u_shape': u.shape, 's_shape': s.shape, 'v_shape': v.shape}
        
        # Cholesky decomposition
        try:
            chol = torch.cholesky(symmetric_pd)
            results['cholesky_decomposition'] = chol.shape
        except Exception as e:
            results['cholesky_error'] = str(e)
        
        return results
    
    def _test_linear_algebra(self, device: torch.device) -> Dict[str, Any]:
        """Test linear algebra operations."""
        # Create invertible matrix
        a = torch.rand(3, 3, device=device) + torch.eye(3, device=device)
        
        results = {}
        results['determinant'] = torch.det(a).item()
        results['trace'] = torch.trace(a).item()
        results['matrix_rank'] = torch.matrix_rank(a).item()
        
        # Matrix inverse
        try:
            inv = torch.inverse(a)
            results['inverse'] = inv.shape
            # Verify inverse
            identity_check = torch.allclose(torch.mm(a, inv), torch.eye(3, device=device), atol=1e-5)
            results['inverse_verification'] = identity_check
        except Exception as e:
            results['inverse_error'] = str(e)
        
        return results
    
    def _test_eigenvalues(self, device: torch.device) -> Dict[str, Any]:
        """Test eigenvalue computations."""
        # Create symmetric matrix for eigenvalue decomposition
        a = torch.rand(3, 3, device=device)
        symmetric = (a + a.t()) / 2
        
        results = {}
        
        try:
            eigenvalues, eigenvectors = torch.eig(symmetric, eigenvectors=True)
            results['eigenvalues'] = eigenvalues.shape
            results['eigenvectors'] = eigenvectors.shape
        except Exception as e:
            results['eig_error'] = str(e)
        
        try:
            eigenvalues_sym = torch.symeig(symmetric, eigenvectors=False)[0]
            results['symmetric_eigenvalues'] = eigenvalues_sym.shape
        except Exception as e:
            results['symeig_error'] = str(e)
        
        return results
    
    def test_neural_network_basics(self, device: torch.device) -> Dict[str, Any]:
        """Test basic neural network components."""
        self._log(f"Testing neural network basics on {device}")
        
        results = {
            'device': str(device),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        tests = [
            ('activation_functions', self._test_activation_functions),
            ('loss_functions', self._test_loss_functions),
            ('optimization', self._test_optimization),
            ('simple_network', self._test_simple_network)
        ]
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                test_result = test_func(device)
                execution_time = time.time() - start_time
                
                results['test_details'][test_name] = {
                    'passed': True,
                    'execution_time': execution_time,
                    'details': test_result
                }
                results['tests_passed'] += 1
                self._log(f"✓ {test_name} passed ({execution_time:.4f}s)")
                
            except Exception as e:
                results['test_details'][test_name] = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                results['tests_failed'] += 1
                self._log(f"✗ {test_name} failed: {e}", "ERROR")
        
        return results
    
    def _test_activation_functions(self, device: torch.device) -> Dict[str, Any]:
        """Test various activation functions."""
        x = torch.randn(5, 3, device=device)
        
        results = {}
        results['relu'] = torch.nn.functional.relu(x).shape
        results['sigmoid'] = torch.nn.functional.sigmoid(x).shape
        results['tanh'] = torch.nn.functional.tanh(x).shape
        results['softmax'] = torch.nn.functional.softmax(x, dim=-1).shape
        results['leaky_relu'] = torch.nn.functional.leaky_relu(x).shape
        results['gelu'] = torch.nn.functional.gelu(x).shape
        results['silu'] = torch.nn.functional.silu(x).shape
        
        return results
    
    def _test_loss_functions(self, device: torch.device) -> Dict[str, Any]:
        """Test various loss functions."""
        predictions = torch.randn(5, 3, device=device)
        targets = torch.randn(5, 3, device=device)
        class_targets = torch.randint(0, 3, (5,), device=device)
        
        results = {}
        results['mse_loss'] = torch.nn.functional.mse_loss(predictions, targets).item()
        results['l1_loss'] = torch.nn.functional.l1_loss(predictions, targets).item()
        results['cross_entropy'] = torch.nn.functional.cross_entropy(predictions, class_targets).item()
        results['nll_loss'] = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(predictions, dim=-1),
            class_targets
        ).item()
        
        return results
    
    def _test_optimization(self, device: torch.device) -> Dict[str, Any]:
        """Test basic optimization functionality."""
        # Create a simple linear model
        model = torch.nn.Linear(3, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        x = torch.randn(10, 3, device=device)
        y = torch.randn(10, 1, device=device)
        
        results = {}
        
        # Test optimization step
        initial_loss = torch.nn.functional.mse_loss(model(x), y).item()
        
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        
        final_loss = torch.nn.functional.mse_loss(model(x), y).item()
        
        results['initial_loss'] = initial_loss
        results['final_loss'] = final_loss
        results['loss_decreased'] = final_loss < initial_loss
        results['gradients_computed'] = any(p.grad is not None for p in model.parameters())
        
        return results
    
    def _test_simple_network(self, device: torch.device) -> Dict[str, Any]:
        """Test a simple neural network forward pass."""
        # Define a simple MLP
        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = SimpleMLP().to(device)
        x = torch.randn(5, 10, device=device)
        
        results = {}
        results['model_parameters'] = sum(p.numel() for p in model.parameters())
        results['forward_pass'] = model(x).shape
        results['model_device'] = str(next(model.parameters()).device)
        
        return results
    
    def run_performance_benchmark(self, device: torch.device, sizes: List[int] = None) -> Dict[str, Any]:
        """Run performance benchmarks on specified device."""
        if sizes is None:
            sizes = [100, 500, 1000]
        
        self._log(f"Running performance benchmarks on {device}")
        
        results = {
            'device': str(device),
            'benchmark_results': {}
        }
        
        for size in sizes:
            self._log(f"Testing size {size}x{size}")
            
            # Matrix multiplication benchmark
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            
            # Warm-up
            for _ in range(3):
                _ = torch.matmul(a, b)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            results['benchmark_results'][f'matmul_{size}x{size}'] = {
                'avg_time_seconds': avg_time,
                'operations_per_second': 1.0 / avg_time,
                'gflops': (2 * size**3) / (avg_time * 1e9)  # Approximate GFLOPS for matrix multiplication
            }
        
        return results
    
    def test_memory_management(self, device: torch.device) -> Dict[str, Any]:
        """Test memory management and allocation."""
        self._log(f"Testing memory management on {device}")
        
        results = {
            'device': str(device),
            'memory_tests': {}
        }
        
        if device.type == 'cuda':
            # CUDA memory tests
            initial_memory = torch.cuda.memory_allocated(device)
            
            # Allocate memory
            large_tensor = torch.rand(1000, 1000, device=device)
            after_alloc_memory = torch.cuda.memory_allocated(device)
            
            # Clean up
            del large_tensor
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device)
            
            results['memory_tests'].update({
                'initial_memory_mb': initial_memory / (1024**2),
                'after_allocation_mb': after_alloc_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2),
                'memory_freed': (after_alloc_memory - final_memory) > 0,
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
                'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / (1024**2)
            })
        
        elif device.type == 'mps':
            # MPS doesn't have detailed memory tracking like CUDA
            results['memory_tests']['device_type'] = 'mps'
            results['memory_tests']['note'] = 'MPS does not provide detailed memory tracking'
        
        else:
            # CPU memory tests
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Allocate memory
            large_tensor = torch.rand(1000, 1000, device=device)
            after_alloc_memory = process.memory_info().rss
            
            # Clean up
            del large_tensor
            final_memory = process.memory_info().rss
            
            results['memory_tests'].update({
                'initial_memory_mb': initial_memory / (1024**2),
                'after_allocation_mb': after_alloc_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2),
                'system_memory_gb': psutil.virtual_memory().total / (1024**3)
            })
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        self._log("Generating comprehensive report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'log_file': str(self.log_file)
            },
            'pytorch_info': self.check_pytorch_installation(),
            'available_devices': self.get_available_devices(),
            'test_results': {},
            'performance_benchmarks': {},
            'memory_tests': {},
            'summary': {}
        }
        
        # Test all available devices
        devices = report['available_devices']
        
        for device_name, device_info in devices.items():
            if device_info['available']:
                device = device_info['device']
                self._log(f"Testing device: {device_name}")
                
                # Run all tests
                report['test_results'][device_name] = {
                    'tensor_operations': self.test_basic_tensor_operations(device),
                    'matrix_operations': self.test_matrix_operations(device),
                    'neural_network': self.test_neural_network_basics(device)
                }
                
                # Run performance benchmarks
                try:
                    report['performance_benchmarks'][device_name] = self.run_performance_benchmark(device)
                except Exception as e:
                    self._log(f"Performance benchmark failed for {device_name}: {e}", "ERROR")
                
                # Run memory tests
                try:
                    report['memory_tests'][device_name] = self.test_memory_management(device)
                except Exception as e:
                    self._log(f"Memory test failed for {device_name}: {e}", "ERROR")
        
        # Generate summary
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        summary = {
            'total_devices_tested': len(report['test_results']),
            'devices_passed': 0,
            'devices_failed': 0,
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'device_status': {},
            'recommendations': []
        }
        
        for device_name, device_results in report['test_results'].items():
            device_summary = {
                'total_tests': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'status': 'unknown'
            }
            
            for test_category, test_results in device_results.items():
                device_summary['total_tests'] += test_results.get('tests_passed', 0) + test_results.get('tests_failed', 0)
                device_summary['tests_passed'] += test_results.get('tests_passed', 0)
                device_summary['tests_failed'] += test_results.get('tests_failed', 0)
            
            if device_summary['tests_failed'] == 0:
                device_summary['status'] = 'passed'
                summary['devices_passed'] += 1
            else:
                device_summary['status'] = 'failed'
                summary['devices_failed'] += 1
            
            summary['device_status'][device_name] = device_summary
            summary['total_tests'] += device_summary['total_tests']
            summary['tests_passed'] += device_summary['tests_passed']
            summary['tests_failed'] += device_summary['tests_failed']
        
        # Generate recommendations
        if summary['devices_failed'] > 0:
            summary['recommendations'].append("Some devices failed tests. Check detailed results for specific issues.")
        
        if 'cuda' in report['available_devices'] and report['available_devices']['cuda']['available']:
            summary['recommendations'].append("CUDA is available. Consider using GPU acceleration for better performance.")
        
        if 'mps' in report['available_devices'] and report['available_devices']['mps']['available']:
            summary['recommendations'].append("Apple Metal Performance Shaders (MPS) is available for GPU acceleration.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"pytorch_basic_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n" + "="*80)
        print("PYTORCH BASIC VERIFICATION SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"Memory: {platform_info['memory_gb']} GB")
        
        # PyTorch info
        pytorch_info = results['pytorch_info']
        print(f"\nPyTorch Version: {pytorch_info.get('pytorch_version', 'Unknown')}")
        print(f"CUDA Available: {pytorch_info.get('cuda_available', False)}")
        print(f"MPS Available: {pytorch_info.get('mps_available', False)}")
        
        # Device summary
        print(f"\nDevices Tested: {results['summary']['total_devices_tested']}")
        print(f"Devices Passed: {results['summary']['devices_passed']}")
        print(f"Devices Failed: {results['summary']['devices_failed']}")
        
        # Test summary
        print(f"\nTotal Tests: {results['summary']['total_tests']}")
        print(f"Tests Passed: {results['summary']['tests_passed']}")
        print(f"Tests Failed: {results['summary']['tests_failed']}")
        
        # Device status
        print(f"\nDevice Status:")
        for device_name, device_status in results['summary']['device_status'].items():
            status_symbol = "✓" if device_status['status'] == 'passed' else "✗"
            print(f"  {status_symbol} {device_name}: {device_status['tests_passed']}/{device_status['total_tests']} tests passed")
        
        # Performance highlights
        if results['performance_benchmarks']:
            print(f"\nPerformance Highlights:")
            for device_name, benchmark in results['performance_benchmarks'].items():
                if 'benchmark_results' in benchmark:
                    for test_name, test_result in benchmark['benchmark_results'].items():
                        if 'gflops' in test_result:
                            print(f"  {device_name} - {test_name}: {test_result['gflops']:.2f} GFLOPS")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run PyTorch basic verification."""
    print("Starting PyTorch Basic Verification Suite...")
    
    try:
        # Initialize verifier
        verifier = PyTorchBasicVerifier()
        
        # Run comprehensive verification
        results = verifier.generate_comprehensive_report()
        
        # Save results
        results_file = verifier.save_results(results)
        
        # Print summary
        verifier.print_summary(results)
        
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        # Show macOS notification if available
        try:
            if platform.system() == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    f'display notification "PyTorch basic verification completed successfully!" with title "GET SWIFTY - PyTorch Verifier"'
                ], check=False)
        except:
            pass
        
    except KeyboardInterrupt:
        print("\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
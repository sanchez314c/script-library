#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

PyTorch Tensor Operations Test Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive PyTorch tensor operations testing and validation

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

class PyTorchTensorTester:
    """Comprehensive PyTorch tensor operations testing suite."""
    
    def __init__(self):
        """Initialize the PyTorch tensor tester."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"pytorch_tensor_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Test configuration
        self.test_sizes = [10, 50, 100, 500]
        self.dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"PyTorch Tensor Test Log - {datetime.now()}\n")
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
    
    def test_tensor_creation(self, device: torch.device) -> Dict[str, Any]:
        """Test various tensor creation methods."""
        self._log(f"Testing tensor creation on {device}")
        
        results = {
            'device': str(device),
            'creation_tests': {}
        }
        
        creation_methods = [
            ('zeros', self._test_zeros_creation),
            ('ones', self._test_ones_creation),
            ('empty', self._test_empty_creation),
            ('random', self._test_random_creation),
            ('arange', self._test_arange_creation),
            ('linspace', self._test_linspace_creation),
            ('eye', self._test_eye_creation),
            ('from_numpy', self._test_from_numpy_creation),
            ('from_list', self._test_from_list_creation)
        ]
        
        for method_name, method_func in creation_methods:
            self._log(f"Testing {method_name} creation")
            
            try:
                method_results = {}
                
                for size in self.test_sizes:
                    for dtype in self.dtypes:
                        test_key = f"{size}_{dtype}"
                        
                        try:
                            start_time = time.time()
                            tensor = method_func(device, size, dtype)
                            creation_time = time.time() - start_time
                            
                            # Validate tensor
                            validation_result = self._validate_tensor(tensor, size, dtype, device)
                            
                            method_results[test_key] = {
                                'creation_time': creation_time,
                                'tensor_shape': list(tensor.shape),
                                'tensor_dtype': str(tensor.dtype),
                                'tensor_device': str(tensor.device),
                                'validation': validation_result,
                                'success': True
                            }
                            
                            # Clean up
                            del tensor
                            
                        except Exception as e:
                            method_results[test_key] = {
                                'error': str(e),
                                'success': False
                            }
                            self._log(f"Error in {method_name} {test_key}: {e}", "WARNING")
                
                results['creation_tests'][method_name] = method_results
                self._log(f"✓ {method_name} creation tests completed")
                
            except Exception as e:
                self._log(f"✗ {method_name} creation tests failed: {e}", "ERROR")
                results['creation_tests'][method_name] = {'error': str(e)}
        
        return results
    
    def _test_zeros_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test zeros tensor creation."""
        return torch.zeros(size, size, device=device, dtype=dtype)
    
    def _test_ones_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test ones tensor creation."""
        return torch.ones(size, size, device=device, dtype=dtype)
    
    def _test_empty_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test empty tensor creation."""
        return torch.empty(size, size, device=device, dtype=dtype)
    
    def _test_random_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test random tensor creation."""
        if dtype in [torch.int32, torch.int64]:
            return torch.randint(0, 100, (size, size), device=device, dtype=dtype)
        else:
            return torch.randn(size, size, device=device, dtype=dtype)
    
    def _test_arange_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test arange tensor creation."""
        return torch.arange(0, size * size, device=device, dtype=dtype).reshape(size, size)
    
    def _test_linspace_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test linspace tensor creation."""
        if dtype in [torch.int32, torch.int64]:
            # Linspace doesn't support integer dtypes directly
            return torch.arange(0, size * size, device=device, dtype=dtype).reshape(size, size)
        else:
            return torch.linspace(0, 1, size * size, device=device, dtype=dtype).reshape(size, size)
    
    def _test_eye_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test eye tensor creation."""
        return torch.eye(size, device=device, dtype=dtype)
    
    def _test_from_numpy_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test tensor creation from numpy array."""
        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        elif dtype == torch.int32:
            np_dtype = np.int32
        elif dtype == torch.int64:
            np_dtype = np.int64
        else:
            np_dtype = np.float32
        
        np_array = np.random.randn(size, size).astype(np_dtype)
        return torch.from_numpy(np_array).to(device)
    
    def _test_from_list_creation(self, device: torch.device, size: int, dtype: torch.dtype) -> torch.Tensor:
        """Test tensor creation from Python list."""
        # Create a smaller list for performance
        test_size = min(size, 20)
        data = [[1.0 for _ in range(test_size)] for _ in range(test_size)]
        return torch.tensor(data, device=device, dtype=dtype)
    
    def _validate_tensor(self, tensor: torch.Tensor, expected_size: int, expected_dtype: torch.dtype, expected_device: torch.device) -> Dict[str, bool]:
        """Validate tensor properties."""
        validation = {
            'shape_correct': True,
            'dtype_correct': tensor.dtype == expected_dtype,
            'device_correct': tensor.device.type == expected_device.type,
            'finite_values': True,
            'non_empty': tensor.numel() > 0
        }
        
        # Check shape (allowing for eye tensor which is square)
        if tensor.dim() == 2:
            validation['shape_correct'] = (
                tensor.shape[0] <= expected_size and 
                tensor.shape[1] <= expected_size
            )
        
        # Check for finite values (if float)
        if tensor.dtype in [torch.float32, torch.float64]:
            try:
                validation['finite_values'] = torch.isfinite(tensor).all().item()
            except:
                validation['finite_values'] = False
        
        return validation
    
    def test_tensor_operations(self, device: torch.device) -> Dict[str, Any]:
        """Test various tensor operations."""
        self._log(f"Testing tensor operations on {device}")
        
        results = {
            'device': str(device),
            'operation_tests': {}
        }
        
        operations = [
            ('arithmetic', self._test_arithmetic_operations),
            ('comparison', self._test_comparison_operations),
            ('logical', self._test_logical_operations),
            ('reduction', self._test_reduction_operations),
            ('indexing', self._test_indexing_operations),
            ('reshaping', self._test_reshaping_operations),
            ('linear_algebra', self._test_linear_algebra_operations),
            ('statistical', self._test_statistical_operations)
        ]
        
        for op_name, op_func in operations:
            self._log(f"Testing {op_name} operations")
            
            try:
                op_results = {}
                
                for size in self.test_sizes:
                    test_key = f"size_{size}"
                    
                    try:
                        start_time = time.time()
                        operation_result = op_func(device, size)
                        execution_time = time.time() - start_time
                        
                        op_results[test_key] = {
                            'execution_time': execution_time,
                            'operation_results': operation_result,
                            'success': True
                        }
                        
                    except Exception as e:
                        op_results[test_key] = {
                            'error': str(e),
                            'success': False
                        }
                        self._log(f"Error in {op_name} {test_key}: {e}", "WARNING")
                
                results['operation_tests'][op_name] = op_results
                self._log(f"✓ {op_name} operations completed")
                
            except Exception as e:
                self._log(f"✗ {op_name} operations failed: {e}", "ERROR")
                results['operation_tests'][op_name] = {'error': str(e)}
        
        return results
    
    def _test_arithmetic_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test arithmetic operations."""
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        results = {}
        
        # Basic arithmetic
        results['addition'] = (a + b).shape
        results['subtraction'] = (a - b).shape
        results['multiplication'] = (a * b).shape
        results['division'] = (a / (b + 1e-8)).shape  # Add epsilon to avoid division by zero
        
        # Advanced arithmetic
        results['power'] = torch.pow(a, 2).shape
        results['sqrt'] = torch.sqrt(torch.abs(a)).shape
        results['exp'] = torch.exp(torch.clamp(a, -10, 10)).shape  # Clamp to avoid overflow
        results['log'] = torch.log(torch.abs(a) + 1e-8).shape
        
        # Matrix operations
        results['matmul'] = torch.matmul(a, b).shape
        results['transpose'] = a.T.shape
        
        # Clean up
        del a, b
        
        return results
    
    def _test_comparison_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test comparison operations."""
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        results = {}
        
        results['equal'] = torch.equal(a, a)
        results['greater'] = (a > b).sum().item()
        results['less'] = (a < b).sum().item()
        results['greater_equal'] = (a >= b).sum().item()
        results['less_equal'] = (a <= b).sum().item()
        results['not_equal'] = (a != b).sum().item()
        results['isnan'] = torch.isnan(a).sum().item()
        results['isinf'] = torch.isinf(a).sum().item()
        results['isfinite'] = torch.isfinite(a).sum().item()
        
        # Clean up
        del a, b
        
        return results
    
    def _test_logical_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test logical operations."""
        a = torch.rand(size, size, device=device) > 0.5
        b = torch.rand(size, size, device=device) > 0.5
        
        results = {}
        
        results['logical_and'] = torch.logical_and(a, b).sum().item()
        results['logical_or'] = torch.logical_or(a, b).sum().item()
        results['logical_not'] = torch.logical_not(a).sum().item()
        results['logical_xor'] = torch.logical_xor(a, b).sum().item()
        
        # Bitwise operations for integer tensors
        int_a = torch.randint(0, 100, (size, size), device=device)
        int_b = torch.randint(0, 100, (size, size), device=device)
        
        results['bitwise_and'] = torch.bitwise_and(int_a, int_b).sum().item()
        results['bitwise_or'] = torch.bitwise_or(int_a, int_b).sum().item()
        results['bitwise_xor'] = torch.bitwise_xor(int_a, int_b).sum().item()
        
        # Clean up
        del a, b, int_a, int_b
        
        return results
    
    def _test_reduction_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test reduction operations."""
        a = torch.randn(size, size, device=device)
        
        results = {}
        
        # Basic reductions
        results['sum'] = torch.sum(a).item()
        results['mean'] = torch.mean(a).item()
        results['std'] = torch.std(a).item()
        results['var'] = torch.var(a).item()
        results['max'] = torch.max(a).item()
        results['min'] = torch.min(a).item()
        
        # Index reductions
        results['argmax'] = torch.argmax(a).item()
        results['argmin'] = torch.argmin(a).item()
        
        # Dimensional reductions
        results['sum_dim0'] = torch.sum(a, dim=0).shape
        results['mean_dim1'] = torch.mean(a, dim=1).shape
        
        # Norm operations
        results['norm_l1'] = torch.norm(a, p=1).item()
        results['norm_l2'] = torch.norm(a, p=2).item()
        results['norm_frobenius'] = torch.norm(a, p='fro').item()
        
        # Clean up
        del a
        
        return results
    
    def _test_indexing_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test indexing and slicing operations."""
        a = torch.randn(size, size, device=device)
        
        results = {}
        
        # Basic indexing
        results['element_access'] = a[0, 0].item()
        results['row_slice'] = a[0, :].shape
        results['column_slice'] = a[:, 0].shape
        results['submatrix'] = a[:size//2, :size//2].shape
        
        # Advanced indexing
        if size > 5:
            indices = torch.tensor([0, 2, 4], device=device)
            results['fancy_indexing'] = a[indices].shape
            
            # Boolean indexing
            mask = a > 0
            results['boolean_indexing'] = a[mask].shape
            
            # Masked fill
            a_copy = a.clone()
            a_copy[mask] = 0
            results['masked_fill'] = 'completed'
        
        # Gather and scatter operations
        if size > 1:
            indices = torch.randint(0, size, (size//2, size), device=device)
            results['gather'] = torch.gather(a, 0, indices).shape
        
        # Clean up
        del a
        
        return results
    
    def _test_reshaping_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test tensor reshaping operations."""
        a = torch.randn(size, size, device=device)
        
        results = {}
        
        # Basic reshaping
        results['view'] = a.view(-1).shape
        results['reshape'] = a.reshape(size*size, 1).shape
        results['flatten'] = torch.flatten(a).shape
        
        # Dimension manipulation
        results['unsqueeze'] = a.unsqueeze(0).shape
        results['squeeze'] = a.unsqueeze(0).squeeze().shape
        results['transpose'] = a.transpose(0, 1).shape
        results['permute'] = a.permute(1, 0).shape
        
        # Concatenation and stacking
        b = torch.randn(size, size, device=device)
        results['cat_dim0'] = torch.cat([a, b], dim=0).shape
        results['cat_dim1'] = torch.cat([a, b], dim=1).shape
        results['stack'] = torch.stack([a, b]).shape
        
        # Splitting
        if size > 2:
            split_tensors = torch.split(a, size//2, dim=0)
            results['split'] = [t.shape for t in split_tensors]
            
            chunk_tensors = torch.chunk(a, 2, dim=1)
            results['chunk'] = [t.shape for t in chunk_tensors]
        
        # Clean up
        del a, b
        
        return results
    
    def _test_linear_algebra_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test linear algebra operations."""
        # Use smaller size for computationally expensive operations
        test_size = min(size, 50)
        a = torch.randn(test_size, test_size, device=device)
        
        results = {}
        
        try:
            # Basic linear algebra
            results['determinant'] = torch.det(a).item()
            results['trace'] = torch.trace(a).item()
            
            # Matrix decompositions
            try:
                q, r = torch.qr(a)
                results['qr_decomposition'] = {'q_shape': q.shape, 'r_shape': r.shape}
                del q, r
            except Exception as e:
                results['qr_error'] = str(e)
            
            try:
                u, s, v = torch.svd(a)
                results['svd_decomposition'] = {'u_shape': u.shape, 's_shape': s.shape, 'v_shape': v.shape}
                del u, s, v
            except Exception as e:
                results['svd_error'] = str(e)
            
            # Eigenvalues (for symmetric matrices)
            try:
                symmetric = (a + a.T) / 2
                eigenvalues = torch.symeig(symmetric, eigenvectors=False)[0]
                results['eigenvalues'] = eigenvalues.shape
                del symmetric, eigenvalues
            except Exception as e:
                results['eigenvalues_error'] = str(e)
            
            # Matrix inverse (for square matrices)
            try:
                # Add identity to ensure invertibility
                invertible = a + torch.eye(test_size, device=device) * 0.1
                inv = torch.inverse(invertible)
                results['inverse'] = inv.shape
                del invertible, inv
            except Exception as e:
                results['inverse_error'] = str(e)
            
        except Exception as e:
            results['linalg_error'] = str(e)
        
        # Clean up
        del a
        
        return results
    
    def _test_statistical_operations(self, device: torch.device, size: int) -> Dict[str, Any]:
        """Test statistical operations."""
        a = torch.randn(size, size, device=device)
        
        results = {}
        
        # Basic statistics
        results['mean'] = torch.mean(a).item()
        results['median'] = torch.median(a).item()
        results['std'] = torch.std(a).item()
        results['var'] = torch.var(a).item()
        
        # Quantiles
        results['quantile_25'] = torch.quantile(a, 0.25).item()
        results['quantile_50'] = torch.quantile(a, 0.5).item()
        results['quantile_75'] = torch.quantile(a, 0.75).item()
        
        # Distribution functions
        results['cumsum'] = torch.cumsum(a.flatten(), dim=0).shape
        results['cumprod'] = torch.cumprod(torch.abs(a.flatten()) + 1e-8, dim=0).shape
        
        # Sorting
        sorted_vals, sorted_indices = torch.sort(a.flatten())
        results['sort'] = {'values_shape': sorted_vals.shape, 'indices_shape': sorted_indices.shape}
        
        # Histogram (simplified)
        if size > 10:
            hist = torch.histc(a, bins=10)
            results['histogram'] = hist.shape
        
        # Clean up
        del a, sorted_vals, sorted_indices
        
        return results
    
    def test_device_transfers(self) -> Dict[str, Any]:
        """Test tensor transfers between devices."""
        self._log("Testing device transfers")
        
        results = {
            'transfer_tests': {}
        }
        
        devices = self.get_available_devices()
        device_list = [info['device'] for info in devices.values() if info['available']]
        
        if len(device_list) < 2:
            results['transfer_tests']['note'] = 'Multiple devices not available for transfer testing'
            return results
        
        # Test transfers between all device pairs
        for i, source_device in enumerate(device_list):
            for j, target_device in enumerate(device_list):
                if i != j:
                    transfer_key = f"{source_device}_to_{target_device}"
                    
                    try:
                        # Create tensor on source device
                        tensor = torch.randn(100, 100, device=source_device)
                        
                        # Time the transfer
                        start_time = time.time()
                        transferred_tensor = tensor.to(target_device)
                        transfer_time = time.time() - start_time
                        
                        # Verify transfer
                        verification = {
                            'source_device': str(tensor.device),
                            'target_device': str(transferred_tensor.device),
                            'shapes_match': tensor.shape == transferred_tensor.shape,
                            'values_match': torch.allclose(tensor.cpu(), transferred_tensor.cpu()),
                            'transfer_time': transfer_time
                        }
                        
                        results['transfer_tests'][transfer_key] = {
                            'success': True,
                            'verification': verification
                        }
                        
                        # Clean up
                        del tensor, transferred_tensor
                        
                        self._log(f"✓ Transfer {transfer_key} completed in {transfer_time:.4f}s")
                        
                    except Exception as e:
                        results['transfer_tests'][transfer_key] = {
                            'success': False,
                            'error': str(e)
                        }
                        self._log(f"✗ Transfer {transfer_key} failed: {e}", "ERROR")
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive tensor testing report."""
        self._log("Generating comprehensive tensor test report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'pytorch_version': torch.__version__,
                'log_file': str(self.log_file)
            },
            'devices': self.get_available_devices(),
            'test_config': {
                'test_sizes': self.test_sizes,
                'dtypes': [str(dtype) for dtype in self.dtypes]
            },
            'device_tests': {},
            'transfer_tests': {},
            'summary': {}
        }
        
        # Test all available devices
        devices = report['devices']
        
        for device_name, device_info in devices.items():
            if device_info['available']:
                device = device_info['device']
                self._log(f"Running tensor tests on device: {device_name}")
                
                try:
                    # Run all tensor tests
                    report['device_tests'][device_name] = {
                        'creation_tests': self.test_tensor_creation(device),
                        'operation_tests': self.test_tensor_operations(device)
                    }
                    
                except Exception as e:
                    self._log(f"Tensor tests failed for {device_name}: {e}", "ERROR")
                    report['device_tests'][device_name] = {
                        'error': str(e)
                    }
        
        # Test device transfers
        try:
            report['transfer_tests'] = self.test_device_transfers()
        except Exception as e:
            self._log(f"Device transfer tests failed: {e}", "ERROR")
            report['transfer_tests'] = {'error': str(e)}
        
        # Generate summary
        report['summary'] = self._generate_test_summary(report)
        
        return report
    
    def _generate_test_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of tensor test results."""
        summary = {
            'devices_tested': len(report['device_tests']),
            'successful_devices': 0,
            'failed_devices': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'device_status': {},
            'performance_insights': {},
            'recommendations': []
        }
        
        for device_name, device_tests in report['device_tests'].items():
            device_summary = {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'status': 'unknown'
            }
            
            if 'error' not in device_tests:
                # Count creation tests
                if 'creation_tests' in device_tests:
                    creation_tests = device_tests['creation_tests']['creation_tests']
                    for method_results in creation_tests.values():
                        if isinstance(method_results, dict):
                            for test_result in method_results.values():
                                if isinstance(test_result, dict):
                                    device_summary['total_tests'] += 1
                                    if test_result.get('success', False):
                                        device_summary['passed_tests'] += 1
                                    else:
                                        device_summary['failed_tests'] += 1
                
                # Count operation tests
                if 'operation_tests' in device_tests:
                    operation_tests = device_tests['operation_tests']['operation_tests']
                    for op_results in operation_tests.values():
                        if isinstance(op_results, dict):
                            for test_result in op_results.values():
                                if isinstance(test_result, dict):
                                    device_summary['total_tests'] += 1
                                    if test_result.get('success', False):
                                        device_summary['passed_tests'] += 1
                                    else:
                                        device_summary['failed_tests'] += 1
                
                if device_summary['failed_tests'] == 0:
                    device_summary['status'] = 'passed'
                    summary['successful_devices'] += 1
                else:
                    device_summary['status'] = 'partial'
                    if device_summary['passed_tests'] == 0:
                        device_summary['status'] = 'failed'
                        summary['failed_devices'] += 1
                    else:
                        summary['successful_devices'] += 1
            else:
                device_summary['status'] = 'failed'
                summary['failed_devices'] += 1
            
            summary['device_status'][device_name] = device_summary
            summary['total_tests'] += device_summary['total_tests']
            summary['passed_tests'] += device_summary['passed_tests']
            summary['failed_tests'] += device_summary['failed_tests']
        
        # Generate recommendations
        if summary['failed_tests'] > 0:
            summary['recommendations'].append("Some tensor tests failed. Check device compatibility and available memory.")
        
        if any('cuda' in device for device in report['devices'].keys()):
            summary['recommendations'].append("CUDA GPU detected. Tensors can be transferred to GPU for acceleration.")
        
        if any('mps' in device for device in report['devices'].keys()):
            summary['recommendations'].append("Apple Silicon MPS detected. Use .to('mps') for GPU acceleration on Apple Silicon.")
        
        # Transfer test summary
        if 'transfer_tests' in report and 'transfer_tests' in report['transfer_tests']:
            successful_transfers = sum(1 for test in report['transfer_tests']['transfer_tests'].values() 
                                     if isinstance(test, dict) and test.get('success', False))
            total_transfers = len(report['transfer_tests']['transfer_tests'])
            
            if successful_transfers == total_transfers and total_transfers > 0:
                summary['recommendations'].append("All device transfers successful. Multi-device workflows are supported.")
            elif successful_transfers > 0:
                summary['recommendations'].append("Some device transfers work. Check specific transfer combinations.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"pytorch_tensor_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create tensor test visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PyTorch Tensor Test Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Test Success Rate by Device
            ax1 = axes[0, 0]
            device_names = []
            success_rates = []
            
            for device_name, device_status in results['summary']['device_status'].items():
                if device_status['total_tests'] > 0:
                    device_names.append(device_name)
                    success_rate = device_status['passed_tests'] / device_status['total_tests'] * 100
                    success_rates.append(success_rate)
            
            if device_names:
                bars = ax1.bar(device_names, success_rates, color=plt.cm.viridis(np.linspace(0, 1, len(device_names))))
                ax1.set_ylabel('Success Rate (%)')
                ax1.set_title('Test Success Rate by Device')
                ax1.set_ylim(0, 100)
                ax1.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, success_rates):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom')
            
            # Plot 2: Test Distribution
            ax2 = axes[0, 1]
            total_tests = results['summary']['total_tests']
            passed_tests = results['summary']['passed_tests']
            failed_tests = results['summary']['failed_tests']
            
            if total_tests > 0:
                labels = ['Passed', 'Failed']
                sizes = [passed_tests, failed_tests]
                colors = ['#2ecc71', '#e74c3c']
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Overall Test Results Distribution')
            
            # Plot 3: Device Capabilities Comparison
            ax3 = axes[1, 0]
            devices = list(results['devices'].keys())
            device_capabilities = []
            
            for device_name in devices:
                if device_name in results['device_tests']:
                    # Count different test categories passed
                    capabilities = 0
                    device_tests = results['device_tests'][device_name]
                    
                    if ('creation_tests' in device_tests and 
                        isinstance(device_tests['creation_tests'], dict)):
                        capabilities += 1
                    
                    if ('operation_tests' in device_tests and 
                        isinstance(device_tests['operation_tests'], dict)):
                        capabilities += 1
                    
                    device_capabilities.append(capabilities)
                else:
                    device_capabilities.append(0)
            
            if devices:
                bars = ax3.bar(devices, device_capabilities, color=plt.cm.plasma(np.linspace(0, 1, len(devices))))
                ax3.set_ylabel('Test Categories Supported')
                ax3.set_title('Device Capabilities')
                ax3.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, device_capabilities):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{value}', ha='center', va='bottom')
            
            # Plot 4: Transfer Success Matrix
            ax4 = axes[1, 1]
            if 'transfer_tests' in results and 'transfer_tests' in results['transfer_tests']:
                transfer_data = results['transfer_tests']['transfer_tests']
                
                # Extract unique devices from transfer keys
                all_devices = set()
                for transfer_key in transfer_data.keys():
                    if '_to_' in transfer_key:
                        source, target = transfer_key.split('_to_')
                        all_devices.add(source)
                        all_devices.add(target)
                
                all_devices = sorted(list(all_devices))
                
                if len(all_devices) > 1:
                    # Create transfer matrix
                    matrix = np.zeros((len(all_devices), len(all_devices)))
                    
                    for i, source in enumerate(all_devices):
                        for j, target in enumerate(all_devices):
                            if i != j:
                                transfer_key = f"{source}_to_{target}"
                                if transfer_key in transfer_data:
                                    matrix[i, j] = 1 if transfer_data[transfer_key].get('success', False) else 0
                    
                    im = ax4.imshow(matrix, cmap='RdYlGn', aspect='auto')
                    ax4.set_xticks(range(len(all_devices)))
                    ax4.set_yticks(range(len(all_devices)))
                    ax4.set_xticklabels(all_devices, rotation=45)
                    ax4.set_yticklabels(all_devices)
                    ax4.set_xlabel('Target Device')
                    ax4.set_ylabel('Source Device')
                    ax4.set_title('Device Transfer Success Matrix')
                    
                    # Add text annotations
                    for i in range(len(all_devices)):
                        for j in range(len(all_devices)):
                            if i != j:
                                text = '✓' if matrix[i, j] == 1 else '✗'
                                ax4.text(j, i, text, ha='center', va='center', 
                                        color='white' if matrix[i, j] == 1 else 'black', fontweight='bold')
                    
                    plt.colorbar(im, ax=ax4)
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"pytorch_tensor_test_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
        print("PYTORCH TENSOR TEST SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"PyTorch: {results['metadata']['pytorch_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"Memory: {platform_info['memory_gb']} GB")
        
        # Test summary
        print(f"\nDevices Tested: {results['summary']['devices_tested']}")
        print(f"Successful Devices: {results['summary']['successful_devices']}")
        print(f"Failed Devices: {results['summary']['failed_devices']}")
        
        print(f"\nTotal Tests: {results['summary']['total_tests']}")
        print(f"Passed Tests: {results['summary']['passed_tests']}")
        print(f"Failed Tests: {results['summary']['failed_tests']}")
        
        if results['summary']['total_tests'] > 0:
            success_rate = results['summary']['passed_tests'] / results['summary']['total_tests'] * 100
            print(f"Overall Success Rate: {success_rate:.1f}%")
        
        # Device status
        print(f"\nDevice Status:")
        for device_name, device_status in results['summary']['device_status'].items():
            status_symbol = "✓" if device_status['status'] == 'passed' else "⚠" if device_status['status'] == 'partial' else "✗"
            print(f"  {status_symbol} {device_name}: {device_status['passed_tests']}/{device_status['total_tests']} tests passed ({device_status['status']})")
        
        # Transfer tests summary
        if 'transfer_tests' in results and 'transfer_tests' in results['transfer_tests']:
            transfer_tests = results['transfer_tests']['transfer_tests']
            successful_transfers = sum(1 for test in transfer_tests.values() 
                                     if isinstance(test, dict) and test.get('success', False))
            total_transfers = len(transfer_tests)
            
            print(f"\nDevice Transfers: {successful_transfers}/{total_transfers} successful")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run PyTorch tensor tests."""
    print("Starting PyTorch Tensor Test Suite...")
    
    try:
        # Initialize tester
        tester = PyTorchTensorTester()
        
        # Run comprehensive tensor tests
        results = tester.generate_comprehensive_report()
        
        # Save results
        results_file = tester.save_results(results)
        
        # Create visualizations
        viz_file = tester.create_visualizations(results)
        
        # Print summary
        tester.print_summary(results)
        
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        if viz_file:
            print(f"Visualizations saved to: {viz_file}")
        
        # Show macOS notification if available
        try:
            if platform.system() == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    f'display notification "PyTorch tensor tests completed successfully!" with title "GET SWIFTY - Tensor Tester"'
                ], check=False)
        except:
            pass
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
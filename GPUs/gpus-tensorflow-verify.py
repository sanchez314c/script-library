#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

TensorFlow GPU Verification & Testing Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive TensorFlow GPU verification and testing

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
        'tensorflow',
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

import tensorflow as tf
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TensorFlowGPUVerifier:
    """Comprehensive TensorFlow GPU verification and testing suite."""
    
    def __init__(self):
        """Initialize the TensorFlow GPU verifier."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"tensorflow_gpu_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # TensorFlow configuration
        self._setup_tensorflow()
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"TensorFlow GPU Verification Log - {datetime.now()}\n")
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
    
    def _setup_tensorflow(self):
        """Setup TensorFlow configuration."""
        try:
            # Reduce TensorFlow logging verbosity
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            
            # Configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            self._log(f"TensorFlow {tf.__version__} configured successfully")
            
        except Exception as e:
            self._log(f"Error configuring TensorFlow: {e}", "WARNING")
    
    def verify_tensorflow_installation(self) -> Dict[str, Any]:
        """Verify TensorFlow installation and configuration."""
        self._log("Verifying TensorFlow installation")
        
        verification = {
            'tensorflow_version': tf.__version__,
            'installation_valid': True,
            'build_info': {},
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__
        }
        
        try:
            # Get build information
            if hasattr(tf.sysconfig, 'get_build_info'):
                build_info = tf.sysconfig.get_build_info()
                verification['build_info'] = dict(build_info)
            
            # Test basic operations
            test_tensor = tf.constant([1, 2, 3, 4])
            result = tf.reduce_sum(test_tensor)
            verification['basic_operations'] = True
            verification['test_result'] = int(result.numpy())
            
            self._log("TensorFlow installation verified successfully")
            
        except Exception as e:
            verification['installation_valid'] = False
            verification['error'] = str(e)
            self._log(f"TensorFlow installation verification failed: {e}", "ERROR")
        
        return verification
    
    def detect_gpu_devices(self) -> Dict[str, Any]:
        """Detect and analyze available GPU devices."""
        self._log("Detecting GPU devices")
        
        detection = {
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_devices': [],
            'mps_devices': [],
            'total_gpu_count': 0,
            'device_details': {}
        }
        
        # Detect CUDA GPUs
        try:
            cuda_gpus = tf.config.list_physical_devices('GPU')
            detection['total_gpu_count'] = len(cuda_gpus)
            
            for i, gpu in enumerate(cuda_gpus):
                gpu_info = {
                    'name': gpu.name,
                    'device_type': gpu.device_type,
                    'index': i
                }
                
                # Get detailed device information
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_info['details'] = dict(details) if details else {}
                    
                    # Extract specific information
                    if details:
                        if 'device_name' in details:
                            gpu_info['device_name'] = details['device_name']
                        if 'compute_capability' in details:
                            gpu_info['compute_capability'] = details['compute_capability']
                        if 'pci_bus_id' in details:
                            gpu_info['pci_bus_id'] = details['pci_bus_id']
                
                except Exception as e:
                    gpu_info['details_error'] = str(e)
                
                detection['gpu_devices'].append(gpu_info)
                detection['device_details'][f'GPU:{i}'] = gpu_info
                
                self._log(f"Detected CUDA GPU {i}: {gpu.name}")
        
        except Exception as e:
            self._log(f"Error detecting CUDA GPUs: {e}", "WARNING")
        
        # Detect MPS devices (Apple Metal)
        try:
            mps_devices = tf.config.list_physical_devices('MPS')
            
            for i, mps in enumerate(mps_devices):
                mps_info = {
                    'name': mps.name,
                    'device_type': mps.device_type,
                    'index': i
                }
                
                detection['mps_devices'].append(mps_info)
                detection['device_details'][f'MPS:{i}'] = mps_info
                
                self._log(f"Detected MPS device {i}: {mps.name}")
        
        except Exception as e:
            self._log(f"MPS devices not available: {e}", "INFO")
        
        total_devices = len(detection['gpu_devices']) + len(detection['mps_devices'])
        self._log(f"Total GPU devices detected: {total_devices}")
        
        return detection
    
    def test_device_functionality(self, device_name: str) -> Dict[str, Any]:
        """Test basic functionality on a specific device."""
        self._log(f"Testing functionality on device: {device_name}")
        
        results = {
            'device': device_name,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        # Define test operations
        tests = [
            ('basic_arithmetic', self._test_basic_arithmetic),
            ('matrix_operations', self._test_matrix_operations),
            ('neural_network', self._test_neural_network),
            ('memory_allocation', self._test_memory_allocation),
            ('data_types', self._test_data_types)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name} test on {device_name}")
            
            try:
                start_time = time.time()
                test_result = test_func(device_name)
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
    
    def _test_basic_arithmetic(self, device_name: str) -> Dict[str, Any]:
        """Test basic arithmetic operations."""
        with tf.device(device_name):
            a = tf.constant([1.0, 2.0, 3.0, 4.0])
            b = tf.constant([2.0, 3.0, 4.0, 5.0])
            
            # Basic operations
            addition = tf.add(a, b)
            multiplication = tf.multiply(a, b)
            division = tf.divide(a, b)
            
            # Reductions
            sum_result = tf.reduce_sum(a)
            mean_result = tf.reduce_mean(a)
            
            return {
                'addition': addition.numpy().tolist(),
                'multiplication': multiplication.numpy().tolist(),
                'division': division.numpy().tolist(),
                'sum': float(sum_result.numpy()),
                'mean': float(mean_result.numpy())
            }
    
    def _test_matrix_operations(self, device_name: str) -> Dict[str, Any]:
        """Test matrix operations."""
        with tf.device(device_name):
            # Create test matrices
            matrix_a = tf.random.normal([100, 100])
            matrix_b = tf.random.normal([100, 100])
            
            # Matrix multiplication
            matmul_result = tf.matmul(matrix_a, matrix_b)
            
            # Transpose
            transpose_result = tf.transpose(matrix_a)
            
            # Eigenvalues (for smaller matrix)
            small_matrix = tf.random.normal([10, 10])
            symmetric_matrix = tf.matmul(small_matrix, small_matrix, transpose_b=True)
            eigenvalues = tf.linalg.eigvals(symmetric_matrix)
            
            return {
                'matmul_shape': matmul_result.shape.as_list(),
                'transpose_shape': transpose_result.shape.as_list(),
                'eigenvalues_shape': eigenvalues.shape.as_list(),
                'eigenvalues_real': tf.reduce_all(tf.math.is_finite(tf.math.real(eigenvalues))).numpy()
            }
    
    def _test_neural_network(self, device_name: str) -> Dict[str, Any]:
        """Test neural network operations."""
        with tf.device(device_name):
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Generate random data
            x_train = tf.random.normal([100, 10])
            y_train = tf.random.uniform([100, 1]) > 0.5
            y_train = tf.cast(y_train, tf.float32)
            
            # Train for one epoch
            history = model.fit(x_train, y_train, epochs=1, verbose=0)
            
            # Make predictions
            predictions = model.predict(x_train[:10], verbose=0)
            
            return {
                'model_parameters': model.count_params(),
                'training_loss': float(history.history['loss'][0]),
                'training_accuracy': float(history.history['accuracy'][0]),
                'predictions_shape': predictions.shape,
                'predictions_range': [float(np.min(predictions)), float(np.max(predictions))]
            }
    
    def _test_memory_allocation(self, device_name: str) -> Dict[str, Any]:
        """Test memory allocation and deallocation."""
        with tf.device(device_name):
            # Allocate progressively larger tensors
            allocation_results = []
            
            sizes = [100, 500, 1000, 2000]
            for size in sizes:
                try:
                    start_time = time.time()
                    tensor = tf.random.normal([size, size])
                    allocation_time = time.time() - start_time
                    
                    # Perform operation to ensure allocation
                    result = tf.reduce_sum(tensor)
                    
                    allocation_results.append({
                        'size': f"{size}x{size}",
                        'allocation_time': allocation_time,
                        'memory_mb': size * size * 4 / (1024**2),  # Approximate for float32
                        'success': True
                    })
                    
                    # Clean up
                    del tensor, result
                    
                except Exception as e:
                    allocation_results.append({
                        'size': f"{size}x{size}",
                        'error': str(e),
                        'success': False
                    })
                    break
            
            return {
                'allocation_tests': allocation_results,
                'max_successful_size': max([r['size'] for r in allocation_results if r['success']], default='None')
            }
    
    def _test_data_types(self, device_name: str) -> Dict[str, Any]:
        """Test different data types."""
        with tf.device(device_name):
            data_type_results = {}
            
            # Test different data types
            dtypes = [
                (tf.float32, 'float32'),
                (tf.float64, 'float64'),
                (tf.int32, 'int32'),
                (tf.int64, 'int64'),
                (tf.bool, 'bool')
            ]
            
            for dtype, dtype_name in dtypes:
                try:
                    if dtype == tf.bool:
                        tensor = tf.constant([True, False, True, False], dtype=dtype)
                        result = tf.reduce_any(tensor)
                    else:
                        tensor = tf.constant([1, 2, 3, 4], dtype=dtype)
                        result = tf.reduce_sum(tensor)
                    
                    data_type_results[dtype_name] = {
                        'supported': True,
                        'tensor_shape': tensor.shape.as_list(),
                        'result_value': str(result.numpy())
                    }
                    
                except Exception as e:
                    data_type_results[dtype_name] = {
                        'supported': False,
                        'error': str(e)
                    }
            
            return data_type_results
    
    def benchmark_device_performance(self, device_name: str) -> Dict[str, Any]:
        """Benchmark performance on a specific device."""
        self._log(f"Benchmarking performance on device: {device_name}")
        
        benchmarks = {
            'device': device_name,
            'performance_tests': {}
        }
        
        # Matrix multiplication benchmark
        with tf.device(device_name):
            sizes = [256, 512, 1024]
            
            for size in sizes:
                try:
                    # Create matrices
                    a = tf.random.normal([size, size])
                    b = tf.random.normal([size, size])
                    
                    # Warmup
                    for _ in range(3):
                        _ = tf.matmul(a, b)
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(10):
                        result = tf.matmul(a, b)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 10
                    gflops = (2 * size**3) / (avg_time * 1e9)
                    
                    benchmarks['performance_tests'][f'matmul_{size}x{size}'] = {
                        'avg_time': avg_time,
                        'gflops': gflops,
                        'success': True
                    }
                    
                    # Clean up
                    del a, b, result
                    
                except Exception as e:
                    benchmarks['performance_tests'][f'matmul_{size}x{size}'] = {
                        'error': str(e),
                        'success': False
                    }
        
        return benchmarks
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive TensorFlow GPU verification report."""
        self._log("Generating comprehensive verification report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'log_file': str(self.log_file)
            },
            'tensorflow_verification': self.verify_tensorflow_installation(),
            'gpu_detection': self.detect_gpu_devices(),
            'device_tests': {},
            'performance_benchmarks': {},
            'summary': {}
        }
        
        # Test all detected devices
        gpu_detection = report['gpu_detection']
        all_devices = []
        
        # Add CUDA devices
        for i, gpu in enumerate(gpu_detection['gpu_devices']):
            all_devices.append(f"GPU:{i}")
        
        # Add MPS devices
        for i, mps in enumerate(gpu_detection['mps_devices']):
            all_devices.append(f"MPS:{i}")
        
        # Add CPU for comparison
        all_devices.append("CPU:0")
        
        # Test each device
        for device_name in all_devices:
            self._log(f"Testing device: {device_name}")
            
            try:
                # Run functionality tests
                device_tests = self.test_device_functionality(device_name)
                report['device_tests'][device_name] = device_tests
                
                # Run performance benchmarks
                performance_benchmarks = self.benchmark_device_performance(device_name)
                report['performance_benchmarks'][device_name] = performance_benchmarks
                
            except Exception as e:
                self._log(f"Device testing failed for {device_name}: {e}", "ERROR")
                report['device_tests'][device_name] = {'error': str(e)}
                report['performance_benchmarks'][device_name] = {'error': str(e)}
        
        # Generate summary
        report['summary'] = self._generate_verification_summary(report)
        
        return report
    
    def _generate_verification_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of verification results."""
        summary = {
            'tensorflow_status': 'unknown',
            'total_devices': 0,
            'working_devices': 0,
            'failed_devices': 0,
            'device_status': {},
            'performance_highlights': {},
            'recommendations': []
        }
        
        # TensorFlow status
        tf_verification = report['tensorflow_verification']
        summary['tensorflow_status'] = 'working' if tf_verification['installation_valid'] else 'failed'
        
        # Device summary
        gpu_detection = report['gpu_detection']
        summary['total_devices'] = (
            len(gpu_detection['gpu_devices']) + 
            len(gpu_detection['mps_devices']) + 1  # +1 for CPU
        )
        
        # Analyze device tests
        for device_name, device_tests in report['device_tests'].items():
            if 'error' not in device_tests:
                if device_tests['tests_failed'] == 0:
                    summary['working_devices'] += 1
                    summary['device_status'][device_name] = 'working'
                else:
                    summary['device_status'][device_name] = 'partial'
                    if device_tests['tests_passed'] == 0:
                        summary['failed_devices'] += 1
                        summary['device_status'][device_name] = 'failed'
                    else:
                        summary['working_devices'] += 1
            else:
                summary['failed_devices'] += 1
                summary['device_status'][device_name] = 'failed'
        
        # Performance highlights
        for device_name, benchmarks in report['performance_benchmarks'].items():
            if 'performance_tests' in benchmarks:
                max_gflops = 0
                for test_name, test_result in benchmarks['performance_tests'].items():
                    if isinstance(test_result, dict) and 'gflops' in test_result:
                        max_gflops = max(max_gflops, test_result['gflops'])
                
                if max_gflops > 0:
                    summary['performance_highlights'][device_name] = {
                        'max_gflops': max_gflops
                    }
        
        # Generate recommendations
        if summary['tensorflow_status'] == 'failed':
            summary['recommendations'].append("TensorFlow installation has issues. Consider reinstalling TensorFlow.")
        
        if len(gpu_detection['gpu_devices']) > 0:
            summary['recommendations'].append("CUDA GPUs detected. TensorFlow can utilize GPU acceleration.")
        
        if len(gpu_detection['mps_devices']) > 0:
            summary['recommendations'].append("Apple Metal Performance Shaders available. Use device='/device:MPS:0' for GPU acceleration.")
        
        if summary['failed_devices'] > 0:
            summary['recommendations'].append("Some devices failed tests. Check GPU drivers and TensorFlow GPU support.")
        
        if summary['working_devices'] > 1:
            summary['recommendations'].append("Multiple working devices available. Consider using tf.distribute for multi-device training.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"tensorflow_gpu_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create TensorFlow GPU verification visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TensorFlow GPU Verification Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Device Status
            ax1 = axes[0, 0]
            device_status = results['summary']['device_status']
            
            if device_status:
                status_counts = {}
                for status in device_status.values():
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                labels = list(status_counts.keys())
                sizes = list(status_counts.values())
                colors = {'working': '#2ecc71', 'partial': '#f39c12', 'failed': '#e74c3c'}
                plot_colors = [colors.get(label, '#95a5a6') for label in labels]
                
                ax1.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Device Status Distribution')
            
            # Plot 2: Performance Comparison
            ax2 = axes[0, 1]
            performance_data = results['summary']['performance_highlights']
            
            if performance_data:
                devices = list(performance_data.keys())
                gflops = [data['max_gflops'] for data in performance_data.values()]
                
                bars = ax2.bar(devices, gflops, color=plt.cm.viridis(np.linspace(0, 1, len(devices))))
                ax2.set_xlabel('Devices')
                ax2.set_ylabel('Peak Performance (GFLOPS)')
                ax2.set_title('Device Performance Comparison')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, gflops):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gflops)*0.01,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # Plot 3: Test Success Rate
            ax3 = axes[1, 0]
            device_tests = results['device_tests']
            
            device_names = []
            success_rates = []
            
            for device_name, tests in device_tests.items():
                if 'error' not in tests and 'tests_passed' in tests and 'tests_failed' in tests:
                    total_tests = tests['tests_passed'] + tests['tests_failed']
                    if total_tests > 0:
                        success_rate = tests['tests_passed'] / total_tests * 100
                        device_names.append(device_name)
                        success_rates.append(success_rate)
            
            if device_names:
                bars = ax3.bar(device_names, success_rates, color=plt.cm.plasma(np.linspace(0, 1, len(device_names))))
                ax3.set_xlabel('Devices')
                ax3.set_ylabel('Test Success Rate (%)')
                ax3.set_title('Test Success Rate by Device')
                ax3.set_ylim(0, 100)
                ax3.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, success_rates):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f'{value:.1f}%', ha='center', va='bottom')
            
            # Plot 4: Hardware Summary
            ax4 = axes[1, 1]
            gpu_detection = results['gpu_detection']
            
            hardware_counts = [
                ('CUDA GPUs', len(gpu_detection['gpu_devices'])),
                ('MPS Devices', len(gpu_detection['mps_devices'])),
                ('CPU Cores', results['metadata']['platform']['cpu_count'])
            ]
            
            labels = [item[0] for item in hardware_counts]
            counts = [item[1] for item in hardware_counts]
            
            bars = ax4.bar(labels, counts, color=['#3498db', '#9b59b6', '#e67e22'])
            ax4.set_ylabel('Count')
            ax4.set_title('Hardware Summary')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{value}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"tensorflow_gpu_verification_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
        print("TENSORFLOW GPU VERIFICATION SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"Memory: {platform_info['memory_gb']} GB")
        
        # TensorFlow info
        tf_verification = results['tensorflow_verification']
        print(f"\nTensorFlow Version: {tf_verification['tensorflow_version']}")
        print(f"Installation Valid: {tf_verification['installation_valid']}")
        
        # GPU detection
        gpu_detection = results['gpu_detection']
        print(f"\nCUDA Available: {gpu_detection['cuda_available']}")
        print(f"CUDA GPUs: {len(gpu_detection['gpu_devices'])}")
        print(f"MPS Devices: {len(gpu_detection['mps_devices'])}")
        
        # Device status
        print(f"\nDevice Status:")
        for device_name, status in results['summary']['device_status'].items():
            status_symbol = "✓" if status == 'working' else "⚠" if status == 'partial' else "✗"
            print(f"  {status_symbol} {device_name}: {status}")
        
        # Performance highlights
        if results['summary']['performance_highlights']:
            print(f"\nPerformance Highlights:")
            for device, perf in results['summary']['performance_highlights'].items():
                print(f"  {device}: {perf['max_gflops']:.2f} GFLOPS")
        
        # Summary stats
        summary = results['summary']
        print(f"\nSummary:")
        print(f"  Total Devices: {summary['total_devices']}")
        print(f"  Working Devices: {summary['working_devices']}")
        print(f"  Failed Devices: {summary['failed_devices']}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run TensorFlow GPU verification."""
    print("Starting TensorFlow GPU Verification Suite...")
    
    try:
        # Initialize verifier
        verifier = TensorFlowGPUVerifier()
        
        # Run comprehensive verification
        results = verifier.generate_comprehensive_report()
        
        # Save results
        results_file = verifier.save_results(results)
        
        # Create visualizations
        viz_file = verifier.create_visualizations(results)
        
        # Print summary
        verifier.print_summary(results)
        
        if results_file:
            print(f"\nDetailed results saved to: {results_file}")
        
        if viz_file:
            print(f"Visualizations saved to: {viz_file}")
        
        # Show macOS notification if available
        try:
            if platform.system() == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    f'display notification "TensorFlow GPU verification completed successfully!" with title "GET SWIFTY - TensorFlow Verifier"'
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
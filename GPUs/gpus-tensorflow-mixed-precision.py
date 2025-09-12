#!/usr/bin/env python3
"""
██████╗ ███████╗████████╗    ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗
██╔════╝ ██╔════╝╚══██╔══╝    ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██║  ███╗█████╗     ██║       ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝ 
██║   ██║██╔══╝     ██║       ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝  
╚██████╔╝███████╗   ██║       ███████║╚███╔███╔╝██║██║        ██║      ██║   
 ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝   

TensorFlow Mixed Precision Performance Suite v1.0.0
Created by: Script.Library
Date: 2025-05-24
Purpose: Comprehensive TensorFlow mixed precision training and optimization analysis

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

class TensorFlowMixedPrecisionBenchmark:
    """Comprehensive TensorFlow mixed precision benchmarking suite."""
    
    def __init__(self):
        """Initialize the TensorFlow mixed precision benchmark suite."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"tensorflow_mixed_precision_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
                f.write(f"TensorFlow Mixed Precision Benchmark Log - {datetime.now()}\n")
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
            # Configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Reduce TensorFlow logging verbosity
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
            
            self._log(f"TensorFlow {tf.__version__} configured successfully")
            
        except Exception as e:
            self._log(f"Error configuring TensorFlow: {e}", "WARNING")
    
    def detect_hardware_capabilities(self) -> Dict[str, Any]:
        """Detect available hardware and capabilities."""
        self._log("Detecting hardware capabilities")
        
        capabilities = {
            'tensorflow_version': tf.__version__,
            'cuda_available': False,
            'cuda_version': None,
            'cudnn_version': None,
            'gpu_devices': [],
            'mixed_precision_supported': False,
            'tensor_cores_available': False
        }
        
        # Check CUDA support
        if tf.test.is_built_with_cuda():
            capabilities['cuda_available'] = True
            
            # Get CUDA version
            if hasattr(tf.sysconfig, 'get_build_info'):
                build_info = tf.sysconfig.get_build_info()
                capabilities['cuda_version'] = build_info.get('cuda_version', 'Unknown')
                capabilities['cudnn_version'] = build_info.get('cudnn_version', 'Unknown')
        
        # Check GPU devices
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(gpu_devices):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                capabilities['gpu_devices'].append({
                    'name': gpu.name,
                    'device_type': gpu.device_type,
                    'details': gpu_details
                })
                
                # Check for Tensor Core support (approximate)
                if gpu_details and 'compute_capability' in gpu_details:
                    compute_capability = gpu_details['compute_capability']
                    # Tensor Cores available on compute capability 7.0+ (V100, RTX series)
                    if compute_capability >= (7, 0):
                        capabilities['tensor_cores_available'] = True
                
        except Exception as e:
            self._log(f"Error detecting GPU devices: {e}", "WARNING")
        
        # Check mixed precision support
        try:
            # Try to create a mixed precision policy
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            capabilities['mixed_precision_supported'] = True
            self._log("Mixed precision policy set successfully")
        except Exception as e:
            self._log(f"Mixed precision not supported: {e}", "WARNING")
            # Fall back to float32
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        return capabilities
    
    def create_test_models(self) -> Dict[str, tf.keras.Model]:
        """Create various test models for benchmarking."""
        self._log("Creating test models")
        
        models = {}
        
        # Simple dense model
        models['dense_model'] = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # Keep output in float32
        ])
        
        # Convolutional model
        models['conv_model'] = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
        ])
        
        # ResNet-like model
        def create_resnet_block(x, filters, stride=1):
            """Create a basic ResNet block."""
            shortcut = x
            
            x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            if stride != 1:
                shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            
            x = tf.keras.layers.Add()([shortcut, x])
            x = tf.keras.layers.ReLU()(x)
            
            return x
        
        # Build ResNet model
        input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(32, 7, strides=2, padding='same')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Add ResNet blocks
        x = create_resnet_block(x, 32)
        x = create_resnet_block(x, 32)
        x = create_resnet_block(x, 64, stride=2)
        x = create_resnet_block(x, 64)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(10, activation='softmax', dtype='float32')(x)
        
        models['resnet_model'] = tf.keras.Model(inputs=input_layer, outputs=output)
        
        # Compile all models
        for name, model in models.items():
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                # Use loss scaling for mixed precision
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self._log(f"Created and compiled {name}")
        
        return models
    
    def generate_synthetic_data(self, dataset_type: str, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic datasets for testing."""
        if dataset_type == 'dense':
            X = np.random.randn(num_samples, 784).astype(np.float32)
            y = np.random.randint(0, 10, num_samples)
        elif dataset_type == 'conv':
            X = np.random.randn(num_samples, 32, 32, 3).astype(np.float32)
            y = np.random.randint(0, 10, num_samples)
        else:
            X = np.random.randn(num_samples, 32, 32, 3).astype(np.float32)
            y = np.random.randint(0, 10, num_samples)
        
        return X, y
    
    def benchmark_precision_comparison(self) -> Dict[str, Any]:
        """Benchmark FP32 vs FP16 mixed precision performance."""
        self._log("Benchmarking precision comparison")
        
        results = {
            'precision_tests': {}
        }
        
        test_configs = [
            {'precision': 'float32', 'policy': 'float32'},
            {'precision': 'mixed_float16', 'policy': 'mixed_float16'}
        ]
        
        models = self.create_test_models()
        
        for config in test_configs:
            precision = config['precision']
            policy_name = config['policy']
            
            self._log(f"Testing {precision} precision")
            
            try:
                # Set precision policy
                policy = tf.keras.mixed_precision.Policy(policy_name)
                tf.keras.mixed_precision.set_global_policy(policy)
                
                precision_results = {}
                
                for model_name, base_model in models.items():
                    self._log(f"Testing {model_name} with {precision}")
                    
                    try:
                        # Recreate model with current precision policy
                        model = tf.keras.models.clone_model(base_model)
                        
                        # Recompile with appropriate optimizer
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                        if precision == 'mixed_float16':
                            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                        
                        model.compile(
                            optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Generate appropriate test data
                        if 'dense' in model_name:
                            X, y = self.generate_synthetic_data('dense', 1000)
                        else:
                            X, y = self.generate_synthetic_data('conv', 1000)
                        
                        # Warmup
                        model.predict(X[:10], verbose=0)
                        
                        # Time training step
                        start_time = time.time()
                        history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
                        training_time = time.time() - start_time
                        
                        # Time inference
                        start_time = time.time()
                        predictions = model.predict(X, verbose=0)
                        inference_time = time.time() - start_time
                        
                        # Memory usage (if CUDA available)
                        memory_info = {}
                        if tf.config.list_physical_devices('GPU'):
                            try:
                                memory_info = {
                                    'peak_memory_mb': tf.config.experimental.get_memory_info('GPU:0')['peak'] / (1024**2)
                                }
                            except:
                                pass
                        
                        precision_results[model_name] = {
                            'training_time': training_time,
                            'inference_time': inference_time,
                            'final_accuracy': float(history.history['accuracy'][-1]),
                            'final_loss': float(history.history['loss'][-1]),
                            'throughput_samples_per_sec': len(X) / inference_time,
                            'memory_info': memory_info,
                            'success': True
                        }
                        
                        self._log(f"✓ {model_name} with {precision} completed")
                        
                        # Clean up
                        del model, X, y, predictions
                        
                    except Exception as e:
                        precision_results[model_name] = {
                            'error': str(e),
                            'success': False
                        }
                        self._log(f"✗ {model_name} with {precision} failed: {e}", "ERROR")
                
                results['precision_tests'][precision] = precision_results
                
            except Exception as e:
                self._log(f"Error testing {precision}: {e}", "ERROR")
                results['precision_tests'][precision] = {'error': str(e)}
        
        return results
    
    def benchmark_batch_sizes(self) -> Dict[str, Any]:
        """Benchmark performance across different batch sizes."""
        self._log("Benchmarking different batch sizes")
        
        results = {
            'batch_size_tests': {}
        }
        
        batch_sizes = [16, 32, 64, 128, 256]
        models = self.create_test_models()
        
        for model_name, model in models.items():
            self._log(f"Testing batch sizes for {model_name}")
            
            model_results = {}
            
            for batch_size in batch_sizes:
                self._log(f"Testing batch size {batch_size}")
                
                try:
                    # Generate appropriate test data
                    if 'dense' in model_name:
                        X, y = self.generate_synthetic_data('dense', 1000)
                    else:
                        X, y = self.generate_synthetic_data('conv', 1000)
                    
                    # Warmup
                    model.predict(X[:batch_size], batch_size=batch_size, verbose=0)
                    
                    # Time training
                    start_time = time.time()
                    history = model.fit(X, y, epochs=2, batch_size=batch_size, verbose=0)
                    training_time = time.time() - start_time
                    
                    # Time inference
                    start_time = time.time()
                    predictions = model.predict(X, batch_size=batch_size, verbose=0)
                    inference_time = time.time() - start_time
                    
                    model_results[f'batch_{batch_size}'] = {
                        'training_time': training_time,
                        'inference_time': inference_time,
                        'throughput_samples_per_sec': len(X) / inference_time,
                        'final_accuracy': float(history.history['accuracy'][-1]),
                        'success': True
                    }
                    
                    self._log(f"✓ Batch size {batch_size} completed")
                    
                    # Clean up
                    del X, y, predictions
                    
                except Exception as e:
                    model_results[f'batch_{batch_size}'] = {
                        'error': str(e),
                        'success': False
                    }
                    self._log(f"✗ Batch size {batch_size} failed: {e}", "ERROR")
            
            results['batch_size_tests'][model_name] = model_results
        
        return results
    
    def benchmark_optimization_techniques(self) -> Dict[str, Any]:
        """Benchmark various optimization techniques."""
        self._log("Benchmarking optimization techniques")
        
        results = {
            'optimization_tests': {}
        }
        
        # Test different optimizers
        optimizers = {
            'adam': tf.keras.optimizers.Adam(learning_rate=0.001),
            'sgd': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=0.001)
        }
        
        # Use the convolutional model for this test
        base_model = self.create_test_models()['conv_model']
        X, y = self.generate_synthetic_data('conv', 1000)
        
        for opt_name, optimizer in optimizers.items():
            self._log(f"Testing {opt_name} optimizer")
            
            try:
                # Clone and compile model with specific optimizer
                model = tf.keras.models.clone_model(base_model)
                
                # Handle mixed precision
                if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                
                model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train model
                start_time = time.time()
                history = model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                training_time = time.time() - start_time
                
                results['optimization_tests'][opt_name] = {
                    'training_time': training_time,
                    'final_accuracy': float(history.history['accuracy'][-1]),
                    'final_loss': float(history.history['loss'][-1]),
                    'convergence_rate': self._calculate_convergence_rate(history.history['loss']),
                    'success': True
                }
                
                self._log(f"✓ {opt_name} optimizer completed")
                
                # Clean up
                del model
                
            except Exception as e:
                results['optimization_tests'][opt_name] = {
                    'error': str(e),
                    'success': False
                }
                self._log(f"✗ {opt_name} optimizer failed: {e}", "ERROR")
        
        # Clean up
        del X, y
        
        return results
    
    def _calculate_convergence_rate(self, loss_history: List[float]) -> float:
        """Calculate convergence rate from loss history."""
        if len(loss_history) < 2:
            return 0.0
        
        # Calculate average rate of loss decrease
        total_decrease = loss_history[0] - loss_history[-1]
        epochs = len(loss_history) - 1
        
        return total_decrease / epochs if epochs > 0 else 0.0
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive mixed precision benchmark report."""
        self._log("Generating comprehensive mixed precision report")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - self.start_time,
                'platform': self.platform_info,
                'log_file': str(self.log_file)
            },
            'hardware_capabilities': self.detect_hardware_capabilities(),
            'benchmark_results': {},
            'summary': {}
        }
        
        # Run all benchmarks
        benchmarks = [
            ('precision_comparison', self.benchmark_precision_comparison),
            ('batch_size_optimization', self.benchmark_batch_sizes),
            ('optimization_techniques', self.benchmark_optimization_techniques)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            self._log(f"Running {benchmark_name} benchmark")
            
            try:
                benchmark_results = benchmark_func()
                report['benchmark_results'][benchmark_name] = benchmark_results
                self._log(f"✓ {benchmark_name} benchmark completed")
                
            except Exception as e:
                self._log(f"✗ {benchmark_name} benchmark failed: {e}", "ERROR")
                report['benchmark_results'][benchmark_name] = {'error': str(e)}
        
        # Generate summary
        report['summary'] = self._generate_benchmark_summary(report)
        
        return report
    
    def _generate_benchmark_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        summary = {
            'hardware_summary': {},
            'performance_insights': {},
            'recommendations': []
        }
        
        # Hardware summary
        hardware = report['hardware_capabilities']
        summary['hardware_summary'] = {
            'tensorflow_version': hardware['tensorflow_version'],
            'cuda_available': hardware['cuda_available'],
            'mixed_precision_supported': hardware['mixed_precision_supported'],
            'tensor_cores_available': hardware['tensor_cores_available'],
            'gpu_count': len(hardware['gpu_devices'])
        }
        
        # Performance insights
        if 'precision_comparison' in report['benchmark_results']:
            precision_data = report['benchmark_results']['precision_comparison']['precision_tests']
            
            if 'float32' in precision_data and 'mixed_float16' in precision_data:
                fp32_results = precision_data['float32']
                fp16_results = precision_data['mixed_float16']
                
                # Calculate speedups
                speedups = {}
                for model_name in fp32_results.keys():
                    if (model_name in fp16_results and 
                        isinstance(fp32_results[model_name], dict) and 
                        isinstance(fp16_results[model_name], dict) and
                        fp32_results[model_name].get('success') and 
                        fp16_results[model_name].get('success')):
                        
                        fp32_time = fp32_results[model_name]['training_time']
                        fp16_time = fp16_results[model_name]['training_time']
                        
                        if fp16_time > 0:
                            speedups[model_name] = fp32_time / fp16_time
                
                summary['performance_insights']['mixed_precision_speedups'] = speedups
        
        # Generate recommendations
        if summary['hardware_summary']['mixed_precision_supported']:
            summary['recommendations'].append("Mixed precision is supported. Use mixed_float16 policy for potential performance gains.")
        
        if summary['hardware_summary']['tensor_cores_available']:
            summary['recommendations'].append("Tensor Cores detected. Mixed precision training will benefit from hardware acceleration.")
        
        if summary['hardware_summary']['cuda_available']:
            summary['recommendations'].append("CUDA GPU detected. Ensure proper batch sizes for optimal GPU utilization.")
        
        if 'mixed_precision_speedups' in summary['performance_insights']:
            avg_speedup = np.mean(list(summary['performance_insights']['mixed_precision_speedups'].values()))
            if avg_speedup > 1.1:
                summary['recommendations'].append(f"Mixed precision provides {avg_speedup:.1f}x average speedup. Consider using for production workloads.")
            elif avg_speedup < 0.9:
                summary['recommendations'].append("Mixed precision may not provide benefits on this hardware. Consider using float32.")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file on desktop."""
        try:
            results_file = self.desktop_path / f"tensorflow_mixed_precision_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self._log(f"Results saved to: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self._log(f"Failed to save results: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create mixed precision benchmark visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TensorFlow Mixed Precision Benchmark Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Precision Comparison
            ax1 = axes[0, 0]
            if 'precision_comparison' in results['benchmark_results']:
                precision_data = results['benchmark_results']['precision_comparison']['precision_tests']
                
                if 'float32' in precision_data and 'mixed_float16' in precision_data:
                    fp32_results = precision_data['float32']
                    fp16_results = precision_data['mixed_float16']
                    
                    models = []
                    fp32_times = []
                    fp16_times = []
                    
                    for model_name in fp32_results.keys():
                        if (model_name in fp16_results and 
                            isinstance(fp32_results[model_name], dict) and 
                            isinstance(fp16_results[model_name], dict) and
                            fp32_results[model_name].get('success') and 
                            fp16_results[model_name].get('success')):
                            
                            models.append(model_name.replace('_model', ''))
                            fp32_times.append(fp32_results[model_name]['training_time'])
                            fp16_times.append(fp16_results[model_name]['training_time'])
                    
                    if models:
                        x_pos = np.arange(len(models))
                        width = 0.35
                        
                        ax1.bar(x_pos - width/2, fp32_times, width, label='Float32', alpha=0.8)
                        ax1.bar(x_pos + width/2, fp16_times, width, label='Mixed Float16', alpha=0.8)
                        
                        ax1.set_xlabel('Models')
                        ax1.set_ylabel('Training Time (seconds)')
                        ax1.set_title('Training Time: FP32 vs Mixed FP16')
                        ax1.set_xticks(x_pos)
                        ax1.set_xticklabels(models)
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
            
            # Plot 2: Speedup Analysis
            ax2 = axes[0, 1]
            if 'mixed_precision_speedups' in results['summary']['performance_insights']:
                speedups = results['summary']['performance_insights']['mixed_precision_speedups']
                
                models = list(speedups.keys())
                speedup_values = list(speedups.values())
                
                colors = ['green' if s > 1.0 else 'red' for s in speedup_values]
                bars = ax2.bar(models, speedup_values, color=colors, alpha=0.7)
                
                ax2.set_xlabel('Models')
                ax2.set_ylabel('Speedup (FP32 time / FP16 time)')
                ax2.set_title('Mixed Precision Speedup')
                ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, speedup_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{value:.2f}x', ha='center', va='bottom')
            
            # Plot 3: Batch Size Performance
            ax3 = axes[1, 0]
            if 'batch_size_optimization' in results['benchmark_results']:
                batch_data = results['benchmark_results']['batch_size_optimization']['batch_size_tests']
                
                # Use first available model
                model_name = list(batch_data.keys())[0] if batch_data else None
                if model_name and isinstance(batch_data[model_name], dict):
                    model_results = batch_data[model_name]
                    
                    batch_sizes = []
                    throughputs = []
                    
                    for batch_key, result in model_results.items():
                        if batch_key.startswith('batch_') and isinstance(result, dict) and result.get('success'):
                            batch_size = int(batch_key.split('_')[1])
                            throughput = result['throughput_samples_per_sec']
                            
                            batch_sizes.append(batch_size)
                            throughputs.append(throughput)
                    
                    if batch_sizes:
                        ax3.plot(batch_sizes, throughputs, marker='o', linewidth=2, markersize=6)
                        ax3.set_xlabel('Batch Size')
                        ax3.set_ylabel('Throughput (samples/sec)')
                        ax3.set_title(f'Throughput vs Batch Size ({model_name})')
                        ax3.grid(True, alpha=0.3)
                        ax3.set_xscale('log', base=2)
            
            # Plot 4: Hardware Capabilities
            ax4 = axes[1, 1]
            hardware = results['hardware_capabilities']
            
            capabilities = []
            labels = []
            
            if hardware['cuda_available']:
                capabilities.append(1)
                labels.append('CUDA')
            
            if hardware['mixed_precision_supported']:
                capabilities.append(1)
                labels.append('Mixed Precision')
            
            if hardware['tensor_cores_available']:
                capabilities.append(1)
                labels.append('Tensor Cores')
            
            if len(hardware['gpu_devices']) > 0:
                capabilities.append(len(hardware['gpu_devices']))
                labels.append('GPU Devices')
            
            if capabilities:
                colors = plt.cm.viridis(np.linspace(0, 1, len(capabilities)))
                bars = ax4.bar(labels, capabilities, color=colors, alpha=0.7)
                
                ax4.set_ylabel('Count / Availability')
                ax4.set_title('Hardware Capabilities')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, capabilities):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{value}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"tensorflow_mixed_precision_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
        print("TENSORFLOW MIXED PRECISION BENCHMARK SUMMARY")
        print("="*80)
        
        # Platform info
        platform_info = results['metadata']['platform']
        print(f"\nPlatform: {platform_info['system']} {platform_info['machine']}")
        print(f"Python: {platform_info['python_version']}")
        print(f"CPU Cores: {platform_info['cpu_count']}")
        print(f"Memory: {platform_info['memory_gb']} GB")
        
        # Hardware capabilities
        hardware = results['hardware_capabilities']
        print(f"\nTensorFlow Version: {hardware['tensorflow_version']}")
        print(f"CUDA Available: {hardware['cuda_available']}")
        print(f"Mixed Precision Supported: {hardware['mixed_precision_supported']}")
        print(f"Tensor Cores Available: {hardware['tensor_cores_available']}")
        print(f"GPU Devices: {len(hardware['gpu_devices'])}")
        
        # Performance insights
        if 'mixed_precision_speedups' in results['summary']['performance_insights']:
            speedups = results['summary']['performance_insights']['mixed_precision_speedups']
            print(f"\nMixed Precision Speedups:")
            for model, speedup in speedups.items():
                print(f"  {model}: {speedup:.2f}x")
            
            avg_speedup = np.mean(list(speedups.values()))
            print(f"  Average Speedup: {avg_speedup:.2f}x")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nExecution Time: {results['metadata']['execution_time']:.2f} seconds")
        print(f"Log File: {results['metadata']['log_file']}")
        print("="*80)

def main():
    """Main function to run TensorFlow mixed precision benchmark."""
    print("Starting TensorFlow Mixed Precision Benchmark Suite...")
    
    try:
        # Initialize benchmark
        benchmark = TensorFlowMixedPrecisionBenchmark()
        
        # Run comprehensive benchmarks
        results = benchmark.generate_comprehensive_report()
        
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
                    f'display notification "TensorFlow mixed precision benchmark completed!" with title "GET SWIFTY - TensorFlow Benchmark"'
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
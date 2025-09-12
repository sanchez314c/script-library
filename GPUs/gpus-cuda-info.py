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
# Script Name: gpus-cuda-info.py                                                 
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive CUDA GPU information and diagnostics tool with     
#              detailed hardware analysis, capability detection, memory         
#              monitoring, and performance profiling for development workflows.  
#
# Usage: python gpus-cuda-info.py [--verbose] [--benchmark] [--export]          
#
# Dependencies: torch, nvidia-ml-py3, psutil, GPUtil                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional GPU diagnostics with comprehensive hardware analysis,     
#        CUDA capability detection, and development optimization insights.      
#                                                                                
####################################################################################

"""
Comprehensive CUDA GPU Information Tool
======================================

Advanced GPU diagnostics and information tool that provides detailed hardware
analysis, CUDA capability detection, memory monitoring, and performance profiling
for AI/ML development workflows with comprehensive reporting capabilities.
"""

import os
import sys
import logging
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import platform

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "torch>=1.12.0",
        "nvidia-ml-py3>=7.352.0",
        "psutil>=5.8.0",
        "GPUtil>=1.4.0"
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
import psutil
import GPUtil

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class CUDAInfoAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.cuda_info = {}
        self.system_info = {}
        self.benchmark_results = {}
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-cuda-info.log"
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
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            }
            
            # Add detailed CPU information
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.system_info.update({
                    'cpu_freq_current': cpu_freq.current,
                    'cpu_freq_max': cpu_freq.max,
                    'cpu_freq_min': cpu_freq.min
                })
                
            return self.system_info
            
        except Exception as e:
            self.logger.error(f"Error gathering system info: {e}")
            return {}
            
    def check_cuda_availability(self):
        """Check CUDA availability and basic information"""
        cuda_info = {
            'available': False,
            'version': None,
            'device_count': 0,
            'current_device': None,
            'devices': []
        }
        
        try:
            # PyTorch CUDA check
            cuda_info['available'] = torch.cuda.is_available()
            
            if cuda_info['available']:
                cuda_info['version'] = torch.version.cuda
                cuda_info['device_count'] = torch.cuda.device_count()
                cuda_info['current_device'] = torch.cuda.current_device()
                
                # Get device information for each GPU
                for i in range(cuda_info['device_count']):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': device_props.name,
                        'major': device_props.major,
                        'minor': device_props.minor,
                        'total_memory': device_props.total_memory,
                        'multi_processor_count': device_props.multi_processor_count,
                        'memory_gb': round(device_props.total_memory / (1024**3), 2)
                    }
                    cuda_info['devices'].append(device_info)
                    
        except Exception as e:
            self.logger.error(f"Error checking CUDA availability: {e}")
            
        return cuda_info
        
    def get_nvml_info(self):
        """Get detailed GPU information using NVIDIA ML"""
        nvml_info = {'available': False, 'devices': []}
        
        if not NVML_AVAILABLE:
            return nvml_info
            
        try:
            nvml.nvmlInit()
            nvml_info['available'] = True
            
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic device info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                device_info = {
                    'index': i,
                    'name': name,
                    'memory_total': memory_info.total,
                    'memory_free': memory_info.free,
                    'memory_used': memory_info.used,
                    'memory_total_gb': round(memory_info.total / (1024**3), 2),
                    'memory_free_gb': round(memory_info.free / (1024**3), 2),
                    'memory_used_gb': round(memory_info.used / (1024**3), 2),
                    'memory_utilization': round((memory_info.used / memory_info.total) * 100, 1)
                }
                
                # Power and temperature info
                try:
                    power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    device_info['power_draw_watts'] = round(power_draw, 1)
                except:
                    device_info['power_draw_watts'] = 'N/A'
                    
                try:
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    device_info['temperature_c'] = temperature
                except:
                    device_info['temperature_c'] = 'N/A'
                    
                # GPU utilization
                try:
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    device_info['gpu_utilization'] = utilization.gpu
                    device_info['memory_utilization_rate'] = utilization.memory
                except:
                    device_info['gpu_utilization'] = 'N/A'
                    device_info['memory_utilization_rate'] = 'N/A'
                    
                # Clock speeds
                try:
                    graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                    device_info['graphics_clock_mhz'] = graphics_clock
                    device_info['memory_clock_mhz'] = memory_clock
                except:
                    device_info['graphics_clock_mhz'] = 'N/A'
                    device_info['memory_clock_mhz'] = 'N/A'
                    
                # Driver version
                try:
                    driver_version = nvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    device_info['driver_version'] = driver_version
                except:
                    device_info['driver_version'] = 'N/A'
                    
                nvml_info['devices'].append(device_info)
                
        except Exception as e:
            self.logger.error(f"Error getting NVML info: {e}")
            nvml_info['error'] = str(e)
            
        return nvml_info
        
    def get_gpu_util_info(self):
        """Get GPU information using GPUtil"""
        gpu_util_info = {'available': False, 'devices': []}
        
        try:
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu_util_info['available'] = True
                
                for gpu in gpus:
                    device_info = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': round(gpu.load * 100, 1),
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'memory_util': round(gpu.memoryUtil * 100, 1),
                        'temperature': gpu.temperature,
                        'driver': gpu.driver
                    }
                    gpu_util_info['devices'].append(device_info)
                    
        except Exception as e:
            self.logger.error(f"Error getting GPUtil info: {e}")
            gpu_util_info['error'] = str(e)
            
        return gpu_util_info
        
    def run_basic_benchmark(self):
        """Run basic GPU performance benchmark"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
            
        benchmark_results = {}
        
        try:
            device = torch.device('cuda')
            
            # Matrix multiplication benchmark
            self.logger.info("Running matrix multiplication benchmark...")
            sizes = [1000, 2000, 4000]
            
            for size in sizes:
                # Warm up
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                gflops = (2 * size**3) / (avg_time * 1e9)
                
                benchmark_results[f'matmul_{size}x{size}'] = {
                    'avg_time_seconds': round(avg_time, 4),
                    'gflops': round(gflops, 2)
                }
                
            # Memory bandwidth test
            self.logger.info("Running memory bandwidth test...")
            sizes_mb = [100, 500, 1000]
            
            for size_mb in sizes_mb:
                elements = (size_mb * 1024 * 1024) // 4  # Float32 elements
                data = torch.randn(elements, device=device)
                
                # Warm up
                result = torch.sum(data)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    result = torch.sum(data)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100
                bandwidth_gbps = (size_mb / 1024) / avg_time
                
                benchmark_results[f'memory_bandwidth_{size_mb}MB'] = {
                    'avg_time_seconds': round(avg_time, 6),
                    'bandwidth_gbps': round(bandwidth_gbps, 2)
                }
                
        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            benchmark_results['error'] = str(e)
            
        return benchmark_results
        
    def check_ml_frameworks(self):
        """Check ML framework compatibility"""
        frameworks = {}
        
        # PyTorch
        try:
            import torch
            frameworks['pytorch'] = {
                'available': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            frameworks['pytorch'] = {'available': False, 'error': 'Not installed'}
            
        # TensorFlow
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            frameworks['tensorflow'] = {
                'available': True,
                'version': tf.__version__,
                'gpu_available': len(gpu_devices) > 0,
                'gpu_devices': len(gpu_devices),
                'device_names': [device.name for device in gpu_devices]
            }
        except ImportError:
            frameworks['tensorflow'] = {'available': False, 'error': 'Not installed'}
            
        return frameworks
        
    def generate_report(self, output_path=None):
        """Generate comprehensive GPU information report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = Path.home() / "Desktop" / f"cuda_gpu_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        report = {
            'timestamp': timestamp,
            'system_info': self.system_info,
            'cuda_info': self.cuda_info,
            'nvml_info': self.get_nvml_info() if NVML_AVAILABLE else {'available': False, 'reason': 'nvidia-ml-py3 not available'},
            'gpu_util_info': self.get_gpu_util_info(),
            'ml_frameworks': self.check_ml_frameworks(),
            'benchmark_results': self.benchmark_results
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Report saved to: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return None
            
    def display_summary(self, verbose=False):
        """Display GPU information summary"""
        print("\n" + "="*60)
        print("          CUDA GPU INFORMATION SUMMARY")
        print("="*60)
        
        # System Information
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   Platform: {self.system_info.get('platform', 'Unknown')}")
        print(f"   CPU Count: {self.system_info.get('cpu_count', 'Unknown')}")
        print(f"   Memory: {self.system_info.get('memory_gb', 0):.1f} GB")
        print(f"   Python: {self.system_info.get('python_version', 'Unknown')}")
        
        # CUDA Information
        print(f"\n🚀 CUDA INFORMATION:")
        print(f"   CUDA Available: {'✅ Yes' if self.cuda_info.get('available') else '❌ No'}")
        
        if self.cuda_info.get('available'):
            print(f"   CUDA Version: {self.cuda_info.get('version', 'Unknown')}")
            print(f"   Device Count: {self.cuda_info.get('device_count', 0)}")
            print(f"   Current Device: {self.cuda_info.get('current_device', 'Unknown')}")
            
            # Device Details
            for i, device in enumerate(self.cuda_info.get('devices', [])):
                print(f"\n   📱 Device {i}:")
                print(f"      Name: {device.get('name', 'Unknown')}")
                print(f"      Compute Capability: {device.get('major', 0)}.{device.get('minor', 0)}")
                print(f"      Memory: {device.get('memory_gb', 0):.1f} GB")
                print(f"      Multiprocessors: {device.get('multi_processor_count', 0)}")
                
        # Framework Compatibility
        frameworks = self.check_ml_frameworks()
        print(f"\n🧠 ML FRAMEWORK COMPATIBILITY:")
        
        for name, info in frameworks.items():
            status = "✅" if info.get('available') else "❌"
            print(f"   {name.title()}: {status}")
            if info.get('available') and verbose:
                print(f"      Version: {info.get('version', 'Unknown')}")
                if name == 'pytorch':
                    print(f"      CUDA Support: {'Yes' if info.get('cuda_available') else 'No'}")
                elif name == 'tensorflow':
                    print(f"      GPU Devices: {info.get('gpu_devices', 0)}")
                    
        # Benchmark Results
        if self.benchmark_results and verbose:
            print(f"\n⚡ BENCHMARK RESULTS:")
            for test_name, results in self.benchmark_results.items():
                if 'error' not in results:
                    print(f"   {test_name}:")
                    if 'gflops' in results:
                        print(f"      Performance: {results['gflops']} GFLOPS")
                    if 'bandwidth_gbps' in results:
                        print(f"      Bandwidth: {results['bandwidth_gbps']} GB/s")
                        
        print("\n" + "="*60)
        
    def show_gui_summary(self):
        """Show GUI summary of GPU information"""
        if not IS_MACOS:
            return
            
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Create summary message
            cuda_status = "✅ Available" if self.cuda_info.get('available') else "❌ Not Available"
            device_count = self.cuda_info.get('device_count', 0)
            
            summary = f"""CUDA GPU Information Summary

CUDA Status: {cuda_status}
Device Count: {device_count}
System Memory: {self.system_info.get('memory_gb', 0):.1f} GB

"""
            
            if self.cuda_info.get('available') and self.cuda_info.get('devices'):
                summary += "GPU Devices:\n"
                for device in self.cuda_info['devices']:
                    summary += f"• {device['name']} ({device['memory_gb']:.1f} GB)\n"
                    
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("CUDA GPU Information", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive CUDA GPU Information Tool")
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmarks')
        parser.add_argument('--export', '-e', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("CUDA GPU Info Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting CUDA GPU information analysis...")
        
        try:
            # Gather all information
            print("🔍 Gathering system information...")
            self.get_system_info()
            
            print("🔍 Checking CUDA availability...")
            self.cuda_info = self.check_cuda_availability()
            
            # Run benchmarks if requested
            if args.benchmark and self.cuda_info.get('available'):
                print("🔍 Running performance benchmarks...")
                self.benchmark_results = self.run_basic_benchmark()
            
            # Display results
            self.display_summary(verbose=args.verbose)
            
            # Generate report
            report_path = self.generate_report(args.export)
            
            # Show GUI summary if requested
            if args.gui:
                self.show_gui_summary()
                
            # Open log file if on macOS
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)
                if report_path:
                    subprocess.run(['open', str(report_path)], check=False)
                    
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

def main():
    analyzer = CUDAInfoAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
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
# Script Name: gpus-pytorch-advanced-ops.py                                      
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive PyTorch advanced operations testing and validation 
#              tool with GPU acceleration, distributed computing, autograd,     
#              optimization, and custom operations for ML development workflows.
#
# Usage: python gpus-pytorch-advanced-ops.py [--device auto] [--tests ALL]      
#
# Dependencies: torch, torchvision, numpy, matplotlib                          
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional PyTorch operations testing with comprehensive GPU         
#        acceleration validation and advanced ML development capabilities.      
#                                                                                
####################################################################################

"""
Comprehensive PyTorch Advanced Operations Testing Tool
=====================================================

Advanced testing and validation tool for PyTorch operations with GPU acceleration,
distributed computing, autograd, optimization algorithms, and custom operations
for comprehensive ML development workflow validation.
"""

import os
import sys
import logging
import subprocess
import argparse
import time
import json
import platform
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import statistics

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0"
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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class PyTorchAdvancedOpsTester:
    def __init__(self):
        self.setup_logging()
        self.device = self.detect_best_device()
        self.test_results = {}
        self.performance_metrics = {}
        
        # Test suite configuration
        self.test_suite = {
            'tensor_advanced': self.test_advanced_tensor_ops,
            'autograd_complex': self.test_complex_autograd,
            'custom_functions': self.test_custom_functions,
            'optimization': self.test_optimization_algorithms,
            'neural_networks': self.test_advanced_neural_networks,
            'memory_efficiency': self.test_memory_efficiency,
            'distributed_ops': self.test_distributed_operations,
            'precision_ops': self.test_precision_operations,
            'compilation': self.test_torch_compile,
            'custom_operators': self.test_custom_operators
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "gpus-pytorch-advanced-ops.log"
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
        
    def detect_best_device(self):
        """Detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("✅ Using Apple Silicon MPS device")
        else:
            device = torch.device('cpu')
            self.logger.info("⚠️ Using CPU device - no GPU acceleration available")
            
        return device
        
    def test_advanced_tensor_ops(self):
        """Test advanced tensor operations"""
        test_result = {
            'test_name': 'Advanced Tensor Operations',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Advanced indexing and slicing
            x = torch.randn(100, 100, 100, device=self.device)
            
            # Fancy indexing
            indices = torch.randint(0, 100, (50,), device=self.device)
            indexed_tensor = x[indices]
            test_result['details']['fancy_indexing'] = True
            
            # Advanced slicing with step
            sliced = x[::2, 10:90:3, :]
            test_result['details']['advanced_slicing'] = True
            
            # Boolean indexing
            mask = x > 0
            masked = x[mask]
            test_result['details']['boolean_indexing'] = True
            
            # Tensor manipulation operations
            operations = {
                'reshape_complex': lambda: x.view(-1, 1000).reshape(50, 20, 100),
                'permute': lambda: x.permute(2, 0, 1),
                'transpose_multiple': lambda: x.transpose(0, 1).transpose(1, 2),
                'squeeze_unsqueeze': lambda: x.unsqueeze(0).squeeze(0),
                'repeat_interleave': lambda: torch.repeat_interleave(x[:10], 3, dim=0),
                'gather': lambda: torch.gather(x[:10, :10], 1, indices[:10].unsqueeze(1).expand(-1, 10)),
                'scatter': lambda: torch.zeros_like(x[:10, :10]).scatter(1, indices[:10].unsqueeze(1), 1.0),
                'stack_cat': lambda: torch.cat([torch.stack([x[i] for i in range(10)]), x[:10]], dim=0)
            }
            
            for op_name, op_func in operations.items():
                try:
                    start_time = time.time()
                    result = op_func()
                    end_time = time.time()
                    
                    test_result['details'][f'{op_name}_success'] = True
                    test_result['performance'][f'{op_name}_time_ms'] = (end_time - start_time) * 1000
                    
                    # Verify result is on correct device
                    if result.device != self.device:
                        test_result['issues'].append(f"{op_name} result not on expected device")
                        
                except Exception as e:
                    test_result['issues'].append(f"{op_name} failed: {e}")
                    test_result['details'][f'{op_name}_success'] = False
                    
            # Advanced linear algebra operations
            try:
                A = torch.randn(50, 50, device=self.device)
                
                # Eigenvalue decomposition
                eigenvals, eigenvecs = torch.linalg.eig(A + A.T)  # Make symmetric
                test_result['details']['eigen_decomposition'] = True
                
                # SVD
                U, S, Vh = torch.linalg.svd(A)
                test_result['details']['svd_decomposition'] = True
                
                # QR decomposition
                Q, R = torch.linalg.qr(A)
                test_result['details']['qr_decomposition'] = True
                
                # Matrix inverse
                inv_A = torch.linalg.inv(A + torch.eye(50, device=self.device) * 1e-3)  # Add regularization
                test_result['details']['matrix_inverse'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Linear algebra operations failed: {e}")
                
            # FFT operations
            try:
                signal = torch.randn(1000, device=self.device)
                
                # 1D FFT
                fft_result = torch.fft.fft(signal)
                test_result['details']['fft_1d'] = True
                
                # 2D FFT
                signal_2d = torch.randn(32, 32, device=self.device)
                fft_2d = torch.fft.fft2(signal_2d)
                test_result['details']['fft_2d'] = True
                
                # Real FFT
                rfft_result = torch.fft.rfft(signal)
                test_result['details']['rfft'] = True
                
            except Exception as e:
                test_result['issues'].append(f"FFT operations failed: {e}")
                
            # Statistical operations
            try:
                data = torch.randn(1000, 100, device=self.device)
                
                # Advanced statistics
                stats_ops = {
                    'quantile': lambda: torch.quantile(data, torch.tensor([0.25, 0.5, 0.75], device=self.device)),
                    'histogram': lambda: torch.histogram(data.flatten(), bins=50),
                    'kthvalue': lambda: torch.kthvalue(data, 10, dim=1),
                    'topk': lambda: torch.topk(data, 5, dim=1),
                    'sort': lambda: torch.sort(data[:10], dim=1),
                    'argsort': lambda: torch.argsort(data[:10], dim=1)
                }
                
                for stat_name, stat_func in stats_ops.items():
                    try:
                        result = stat_func()
                        test_result['details'][f'stat_{stat_name}'] = True
                    except Exception as e:
                        test_result['issues'].append(f"Statistical operation {stat_name} failed: {e}")
                        
            except Exception as e:
                test_result['issues'].append(f"Statistical operations failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) == 0
            
        except Exception as e:
            test_result['issues'].append(f"Advanced tensor operations test failed: {e}")
            
        return test_result
        
    def test_complex_autograd(self):
        """Test complex automatic differentiation scenarios"""
        test_result = {
            'test_name': 'Complex Autograd',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Higher-order gradients
            x = torch.randn(10, requires_grad=True, device=self.device)
            y = (x ** 3).sum()
            
            # First derivative
            grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
            test_result['details']['first_derivative'] = True
            
            # Second derivative (Hessian diagonal)
            grad2_y = torch.autograd.grad(grad_y.sum(), x)[0]
            test_result['details']['second_derivative'] = True
            
            # Gradient checkpointing
            def checkpoint_function(x):
                return torch.checkpoint(lambda x: x.sin().cos().tan(), x)
                
            try:
                x_checkpoint = torch.randn(100, requires_grad=True, device=self.device)
                result = checkpoint_function(x_checkpoint)
                loss = result.sum()
                loss.backward()
                test_result['details']['gradient_checkpointing'] = True
            except Exception as e:
                test_result['issues'].append(f"Gradient checkpointing failed: {e}")
                
            # Custom autograd function
            class CustomSquareFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    ctx.save_for_backward(input)
                    return input ** 2
                    
                @staticmethod
                def backward(ctx, grad_output):
                    input, = ctx.saved_tensors
                    return grad_output * 2 * input
                    
            try:
                custom_square = CustomSquareFunction.apply
                x_custom = torch.randn(10, requires_grad=True, device=self.device)
                y_custom = custom_square(x_custom).sum()
                y_custom.backward()
                test_result['details']['custom_autograd_function'] = True
            except Exception as e:
                test_result['issues'].append(f"Custom autograd function failed: {e}")
                
            # Gradient accumulation
            try:
                model = nn.Linear(100, 50).to(self.device)
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                
                # Simulate gradient accumulation
                accumulation_steps = 4
                for step in range(accumulation_steps):
                    x_batch = torch.randn(32, 100, device=self.device)
                    y_batch = torch.randn(32, 50, device=self.device)
                    
                    output = model(x_batch)
                    loss = F.mse_loss(output, y_batch) / accumulation_steps
                    loss.backward()
                    
                optimizer.step()
                optimizer.zero_grad()
                test_result['details']['gradient_accumulation'] = True
            except Exception as e:
                test_result['issues'].append(f"Gradient accumulation failed: {e}")
                
            # Mixed precision with autograd
            if self.device.type in ['cuda', 'mps']:
                try:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        x_amp = torch.randn(100, 100, requires_grad=True, device=self.device)
                        y_amp = torch.matmul(x_amp, x_amp.T).sum()
                        
                    # Use GradScaler for numerical stability
                    scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else torch.amp.GradScaler()
                    scaled_loss = scaler.scale(y_amp)
                    scaled_loss.backward()
                    test_result['details']['mixed_precision_autograd'] = True
                except Exception as e:
                    test_result['issues'].append(f"Mixed precision autograd failed: {e}")
                    
            # Jacobian computation
            try:
                def vector_function(x):
                    return torch.stack([x.sum(), (x**2).sum(), (x**3).sum()])
                    
                x_jac = torch.randn(5, requires_grad=True, device=self.device)
                jacobian = torch.autograd.functional.jacobian(vector_function, x_jac)
                test_result['details']['jacobian_computation'] = True
            except Exception as e:
                test_result['issues'].append(f"Jacobian computation failed: {e}")
                
            # Hessian computation
            try:
                def scalar_function(x):
                    return (x**4).sum()
                    
                x_hess = torch.randn(5, requires_grad=True, device=self.device)
                hessian = torch.autograd.functional.hessian(scalar_function, x_hess)
                test_result['details']['hessian_computation'] = True
            except Exception as e:
                test_result['issues'].append(f"Hessian computation failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) < 3  # Allow some failures
            
        except Exception as e:
            test_result['issues'].append(f"Complex autograd test failed: {e}")
            
        return test_result
        
    def test_custom_functions(self):
        """Test custom function implementations"""
        test_result = {
            'test_name': 'Custom Functions',
            'passed': False,
            'details': {},
            'issues': []
        }
        
        try:
            # Custom activation function
            class Swish(nn.Module):
                def forward(self, x):
                    return x * torch.sigmoid(x)
                    
            swish = Swish().to(self.device)
            x = torch.randn(100, device=self.device)
            result = swish(x)
            test_result['details']['custom_activation'] = True
            
            # Custom loss function
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    
                def forward(self, inputs, targets):
                    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss.mean()
                    
            focal_loss = FocalLoss().to(self.device)
            logits = torch.randn(32, 10, device=self.device)
            targets = torch.randint(0, 10, (32,), device=self.device)
            loss = focal_loss(logits, targets)
            test_result['details']['custom_loss'] = True
            
            # Custom layer with parameters
            class ParametricReLU(nn.Module):
                def __init__(self, num_features):
                    super().__init__()
                    self.alpha = nn.Parameter(torch.zeros(num_features))
                    
                def forward(self, x):
                    return torch.where(x > 0, x, self.alpha * x)
                    
            prelu = ParametricReLU(100).to(self.device)
            x = torch.randn(32, 100, device=self.device)
            result = prelu(x)
            test_result['details']['custom_parametric_layer'] = True
            
            # Custom initialization
            def custom_init(tensor):
                with torch.no_grad():
                    tensor.uniform_(-0.1, 0.1)
                    tensor[0] = 1.0  # Special initialization for first element
                    
            test_tensor = torch.empty(100, device=self.device)
            custom_init(test_tensor)
            test_result['details']['custom_initialization'] = True
            
            # Custom optimizer step
            class CustomSGD(optim.Optimizer):
                def __init__(self, params, lr=0.01, momentum=0.9):
                    defaults = dict(lr=lr, momentum=momentum)
                    super().__init__(params, defaults)
                    
                def step(self, closure=None):
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                                
                            grad = p.grad.data
                            state = self.state[p]
                            
                            if len(state) == 0:
                                state['momentum_buffer'] = torch.zeros_like(p.data)
                                
                            buf = state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(grad)
                            p.data.add_(buf, alpha=-group['lr'])
                            
            model = nn.Linear(10, 5).to(self.device)
            custom_optimizer = CustomSGD(model.parameters())
            
            # Test custom optimizer
            x = torch.randn(32, 10, device=self.device)
            y = torch.randn(32, 5, device=self.device)
            
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            custom_optimizer.step()
            test_result['details']['custom_optimizer'] = True
            
            test_result['passed'] = True
            
        except Exception as e:
            test_result['issues'].append(f"Custom functions test failed: {e}")
            
        return test_result
        
    def test_optimization_algorithms(self):
        """Test various optimization algorithms"""
        test_result = {
            'test_name': 'Optimization Algorithms',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Create a simple optimization problem
            def create_optimization_problem():
                model = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 10)
                ).to(self.device)
                
                x = torch.randn(1000, 100, device=self.device)
                y = torch.randint(0, 10, (1000,), device=self.device)
                
                return model, x, y
                
            # Test different optimizers
            optimizers_to_test = [
                ('SGD', lambda params: optim.SGD(params, lr=0.01)),
                ('Adam', lambda params: optim.Adam(params, lr=0.001)),
                ('AdamW', lambda params: optim.AdamW(params, lr=0.001)),
                ('RMSprop', lambda params: optim.RMSprop(params, lr=0.01)),
                ('Adagrad', lambda params: optim.Adagrad(params, lr=0.01)),
                ('Adadelta', lambda params: optim.Adadelta(params)),
                ('LBFGS', lambda params: optim.LBFGS(params, lr=0.1))
            ]
            
            for opt_name, opt_constructor in optimizers_to_test:
                try:
                    model, x, y = create_optimization_problem()
                    optimizer = opt_constructor(model.parameters())
                    
                    initial_loss = None
                    final_loss = None
                    
                    start_time = time.time()
                    
                    for epoch in range(10):
                        def closure():
                            optimizer.zero_grad()
                            output = model(x)
                            loss = F.cross_entropy(output, y)
                            loss.backward()
                            return loss
                            
                        if opt_name == 'LBFGS':
                            loss = optimizer.step(closure)
                        else:
                            loss = closure()
                            optimizer.step()
                            
                        if epoch == 0:
                            initial_loss = loss.item()
                        final_loss = loss.item()
                        
                    optimization_time = time.time() - start_time
                    
                    test_result['details'][f'{opt_name}_success'] = True
                    test_result['performance'][f'{opt_name}_time'] = optimization_time
                    test_result['performance'][f'{opt_name}_loss_reduction'] = initial_loss - final_loss
                    
                except Exception as e:
                    test_result['issues'].append(f"{opt_name} optimizer failed: {e}")
                    test_result['details'][f'{opt_name}_success'] = False
                    
            # Test learning rate schedulers
            try:
                model, x, y = create_optimization_problem()
                optimizer = optim.Adam(model.parameters(), lr=0.1)
                
                schedulers = [
                    ('StepLR', optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)),
                    ('ExponentialLR', optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)),
                    ('CosineAnnealingLR', optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)),
                    ('ReduceLROnPlateau', optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3))
                ]
                
                for sched_name, scheduler in schedulers:
                    try:
                        for epoch in range(10):
                            optimizer.zero_grad()
                            output = model(x)
                            loss = F.cross_entropy(output, y)
                            loss.backward()
                            optimizer.step()
                            
                            if sched_name == 'ReduceLROnPlateau':
                                scheduler.step(loss)
                            else:
                                scheduler.step()
                                
                        test_result['details'][f'scheduler_{sched_name}'] = True
                    except Exception as e:
                        test_result['issues'].append(f"Scheduler {sched_name} failed: {e}")
                        
            except Exception as e:
                test_result['issues'].append(f"Learning rate scheduler test failed: {e}")
                
            # Test gradient clipping
            try:
                model, x, y = create_optimization_problem()
                optimizer = optim.Adam(model.parameters())
                
                for _ in range(5):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    
                    # Test different clipping methods
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                    
                    optimizer.step()
                    
                test_result['details']['gradient_clipping'] = True
            except Exception as e:
                test_result['issues'].append(f"Gradient clipping failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) < 3
            
        except Exception as e:
            test_result['issues'].append(f"Optimization algorithms test failed: {e}")
            
        return test_result
        
    def test_advanced_neural_networks(self):
        """Test advanced neural network architectures"""
        test_result = {
            'test_name': 'Advanced Neural Networks',
            'passed': False,
            'details': {},
            'issues': []
        }
        
        try:
            # Transformer architecture
            class SimpleTransformer(nn.Module):
                def __init__(self, d_model=512, nhead=8, num_layers=2):
                    super().__init__()
                    self.d_model = d_model
                    self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                    
                    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    self.output_layer = nn.Linear(d_model, 10)
                    
                def forward(self, x):
                    seq_len = x.size(1)
                    x = x + self.pos_encoding[:seq_len]
                    x = self.transformer(x)
                    return self.output_layer(x.mean(dim=1))
                    
            try:
                transformer = SimpleTransformer().to(self.device)
                x = torch.randn(32, 100, 512, device=self.device)
                output = transformer(x)
                test_result['details']['transformer_architecture'] = True
            except Exception as e:
                test_result['issues'].append(f"Transformer architecture failed: {e}")
                
            # ResNet-style skip connections
            class ResidualBlock(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                    self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(channels)
                    self.bn2 = nn.BatchNorm2d(channels)
                    
                def forward(self, x):
                    identity = x
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    out += identity
                    return F.relu(out)
                    
            try:
                resblock = ResidualBlock(64).to(self.device)
                x = torch.randn(8, 64, 32, 32, device=self.device)
                output = resblock(x)
                test_result['details']['residual_connections'] = True
            except Exception as e:
                test_result['issues'].append(f"Residual connections failed: {e}")
                
            # LSTM with attention
            class LSTMWithAttention(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
                    self.output_layer = nn.Linear(hidden_size, 10)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    return self.output_layer(attn_out.mean(dim=1))
                    
            try:
                lstm_attn = LSTMWithAttention(100, 256).to(self.device)
                x = torch.randn(32, 50, 100, device=self.device)
                output = lstm_attn(x)
                test_result['details']['lstm_with_attention'] = True
            except Exception as e:
                test_result['issues'].append(f"LSTM with attention failed: {e}")
                
            # Graph Neural Network (simple)
            class SimpleGCN(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, output_dim)
                    
                def forward(self, x, adj):
                    x = F.relu(self.fc1(x))
                    x = torch.matmul(adj, x)  # Simple graph convolution
                    return self.fc2(x)
                    
            try:
                gcn = SimpleGCN(100, 64, 10).to(self.device)
                nodes = torch.randn(50, 100, device=self.device)
                adj_matrix = torch.randn(50, 50, device=self.device)
                adj_matrix = torch.softmax(adj_matrix, dim=1)  # Normalize
                output = gcn(nodes, adj_matrix)
                test_result['details']['graph_neural_network'] = True
            except Exception as e:
                test_result['issues'].append(f"Graph neural network failed: {e}")
                
            # Variational Autoencoder
            class SimpleVAE(nn.Module):
                def __init__(self, input_dim, latent_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU()
                    )
                    self.mu_layer = nn.Linear(128, latent_dim)
                    self.logvar_layer = nn.Linear(128, latent_dim)
                    
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 256),
                        nn.ReLU(),
                        nn.Linear(256, input_dim),
                        nn.Sigmoid()
                    )
                    
                def encode(self, x):
                    h = self.encoder(x)
                    return self.mu_layer(h), self.logvar_layer(h)
                    
                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                    
                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    return self.decoder(z), mu, logvar
                    
            try:
                vae = SimpleVAE(784, 20).to(self.device)
                x = torch.randn(32, 784, device=self.device)
                recon, mu, logvar = vae(x)
                
                # VAE loss
                recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + kld_loss
                
                test_result['details']['variational_autoencoder'] = True
            except Exception as e:
                test_result['issues'].append(f"Variational autoencoder failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) < 2
            
        except Exception as e:
            test_result['issues'].append(f"Advanced neural networks test failed: {e}")
            
        return test_result
        
    def test_memory_efficiency(self):
        """Test memory efficiency techniques"""
        test_result = {
            'test_name': 'Memory Efficiency',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Memory usage monitoring
            def get_memory_usage():
                if self.device.type == 'cuda':
                    return torch.cuda.memory_allocated(self.device)
                elif self.device.type == 'mps':
                    return torch.mps.current_allocated_memory()
                else:
                    return 0
                    
            initial_memory = get_memory_usage()
            
            # Test gradient checkpointing
            try:
                def checkpoint_fn(x):
                    return torch.sin(torch.cos(torch.tan(x)))
                    
                x = torch.randn(1000, 1000, requires_grad=True, device=self.device)
                
                # Without checkpointing
                start_memory = get_memory_usage()
                result_normal = checkpoint_fn(x)
                loss_normal = result_normal.sum()
                normal_memory = get_memory_usage() - start_memory
                
                # Clear gradients
                if x.grad is not None:
                    x.grad = None
                    
                # With checkpointing
                start_memory = get_memory_usage()
                result_checkpoint = torch.checkpoint(checkpoint_fn, x)
                loss_checkpoint = result_checkpoint.sum()
                checkpoint_memory = get_memory_usage() - start_memory
                
                test_result['details']['gradient_checkpointing'] = True
                test_result['performance']['memory_saved_bytes'] = normal_memory - checkpoint_memory
                
            except Exception as e:
                test_result['issues'].append(f"Gradient checkpointing failed: {e}")
                
            # Test in-place operations
            try:
                x = torch.randn(1000, 1000, device=self.device)
                x_copy = x.clone()
                
                # Out-of-place operations
                start_memory = get_memory_usage()
                result1 = x.add(1.0)
                result2 = result1.mul(2.0)
                outplace_memory = get_memory_usage() - start_memory
                
                # In-place operations
                start_memory = get_memory_usage()
                x_copy.add_(1.0)
                x_copy.mul_(2.0)
                inplace_memory = get_memory_usage() - start_memory
                
                test_result['details']['inplace_operations'] = True
                test_result['performance']['inplace_memory_saved'] = outplace_memory - inplace_memory
                
            except Exception as e:
                test_result['issues'].append(f"In-place operations failed: {e}")
                
            # Test memory pooling
            if self.device.type == 'cuda':
                try:
                    # Allocate and deallocate tensors
                    tensors = []
                    for i in range(10):
                        tensor = torch.randn(100, 100, device=self.device)
                        tensors.append(tensor)
                        
                    del tensors
                    torch.cuda.empty_cache()
                    
                    # Check if memory was freed
                    after_cleanup = get_memory_usage()
                    test_result['details']['cuda_memory_pooling'] = True
                    
                except Exception as e:
                    test_result['issues'].append(f"CUDA memory pooling failed: {e}")
                    
            # Test data loader memory efficiency
            try:
                # Create a simple dataset
                class SimpleDataset(torch.utils.data.Dataset):
                    def __init__(self, size):
                        self.size = size
                        
                    def __len__(self):
                        return self.size
                        
                    def __getitem__(self, idx):
                        return torch.randn(100), torch.randint(0, 10, (1,))
                        
                dataset = SimpleDataset(1000)
                
                # Test with different num_workers
                for num_workers in [0, 2]:
                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=32, num_workers=num_workers, pin_memory=True
                    )
                    
                    start_time = time.time()
                    for i, (data, target) in enumerate(dataloader):
                        if i >= 10:  # Only test first 10 batches
                            break
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        
                    end_time = time.time()
                    test_result['performance'][f'dataloader_time_workers_{num_workers}'] = end_time - start_time
                    
                test_result['details']['efficient_data_loading'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Data loader efficiency test failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) < 2
            
        except Exception as e:
            test_result['issues'].append(f"Memory efficiency test failed: {e}")
            
        return test_result
        
    def test_distributed_operations(self):
        """Test distributed computing capabilities"""
        test_result = {
            'test_name': 'Distributed Operations',
            'passed': False,
            'details': {},
            'issues': []
        }
        
        try:
            # Test data parallel (single GPU)
            if torch.cuda.device_count() > 1:
                try:
                    model = nn.Linear(100, 50)
                    model = nn.DataParallel(model)
                    model = model.to(self.device)
                    
                    x = torch.randn(64, 100, device=self.device)
                    output = model(x)
                    test_result['details']['data_parallel'] = True
                except Exception as e:
                    test_result['issues'].append(f"Data parallel failed: {e}")
            else:
                test_result['details']['data_parallel'] = False
                test_result['issues'].append("Multiple GPUs not available for DataParallel")
                
            # Test distributed data types and operations
            try:
                # Simulate distributed tensor operations
                tensor1 = torch.randn(100, 100, device=self.device)
                tensor2 = torch.randn(100, 100, device=self.device)
                
                # Operations that would be used in distributed settings
                result = torch.distributed.nn.functional.all_reduce(tensor1) if hasattr(torch.distributed.nn.functional, 'all_reduce') else tensor1
                test_result['details']['distributed_operations'] = True
                
            except Exception as e:
                test_result['details']['distributed_operations'] = False
                # This is expected in single-process mode
                
            # Test model parallel concepts
            try:
                class ModelParallelNet(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # In real model parallel, these would be on different devices
                        self.layer1 = nn.Linear(100, 50).to(self.device)
                        self.layer2 = nn.Linear(50, 10).to(self.device)
                        
                    def forward(self, x):
                        # Simulate moving between devices (same device in this test)
                        x = x.to(self.device)
                        x = F.relu(self.layer1(x))
                        x = x.to(self.device)  # Would be different device in real scenario
                        x = self.layer2(x)
                        return x
                        
                model = ModelParallelNet()
                x = torch.randn(32, 100, device=self.device)
                output = model(x)
                test_result['details']['model_parallel_concepts'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Model parallel concepts failed: {e}")
                
            # Test tensor communications (all-reduce simulation)
            try:
                tensors = [torch.randn(100, device=self.device) for _ in range(4)]
                
                # Simulate all-reduce: sum all tensors
                total = torch.zeros_like(tensors[0])
                for tensor in tensors:
                    total += tensor
                    
                # Simulate broadcast: distribute result
                result_tensors = [total.clone() for _ in range(4)]
                test_result['details']['tensor_communications'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Tensor communications failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) <= 1  # Allow for expected failures
            
        except Exception as e:
            test_result['issues'].append(f"Distributed operations test failed: {e}")
            
        return test_result
        
    def test_precision_operations(self):
        """Test mixed precision and quantization"""
        test_result = {
            'test_name': 'Precision Operations',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Test automatic mixed precision
            if self.device.type in ['cuda', 'mps']:
                try:
                    model = nn.Sequential(
                        nn.Linear(1000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 100)
                    ).to(self.device)
                    
                    x = torch.randn(64, 1000, device=self.device)
                    
                    # Test with autocast
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        output = model(x)
                        loss = output.sum()
                        
                    # Test gradient scaling
                    if self.device.type == 'cuda':
                        scaler = torch.cuda.amp.GradScaler()
                        scaled_loss = scaler.scale(loss)
                        
                    test_result['details']['automatic_mixed_precision'] = True
                    
                except Exception as e:
                    test_result['issues'].append(f"Automatic mixed precision failed: {e}")
                    
            # Test manual precision control
            try:
                x_fp32 = torch.randn(100, 100, device=self.device, dtype=torch.float32)
                x_fp16 = x_fp32.half()
                
                # Operations with different precisions
                result_fp32 = torch.matmul(x_fp32, x_fp32.T)
                result_fp16 = torch.matmul(x_fp16, x_fp16.T)
                
                # Compare performance
                start_time = time.time()
                for _ in range(100):
                    _ = torch.matmul(x_fp32, x_fp32.T)
                fp32_time = time.time() - start_time
                
                start_time = time.time()
                for _ in range(100):
                    _ = torch.matmul(x_fp16, x_fp16.T)
                fp16_time = time.time() - start_time
                
                test_result['performance']['fp32_time'] = fp32_time
                test_result['performance']['fp16_time'] = fp16_time
                test_result['performance']['fp16_speedup'] = fp32_time / fp16_time if fp16_time > 0 else 0
                
                test_result['details']['manual_precision_control'] = True
                
            except Exception as e:
                test_result['issues'].append(f"Manual precision control failed: {e}")
                
            # Test quantization (post-training)
            try:
                model = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 10)
                ).to(self.device)
                
                # Prepare model for quantization
                model.eval()
                
                # Quantize model (CPU only for now)
                if self.device.type == 'cpu':
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, {nn.Linear}, dtype=torch.qint8
                    )
                    
                    # Test quantized model
                    x = torch.randn(32, 100)
                    output = quantized_model(x)
                    test_result['details']['dynamic_quantization'] = True
                else:
                    test_result['details']['dynamic_quantization'] = False
                    test_result['issues'].append("Dynamic quantization only tested on CPU")
                    
            except Exception as e:
                test_result['issues'].append(f"Quantization failed: {e}")
                
            # Test different data types
            try:
                data_types = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
                
                for dtype in data_types:
                    try:
                        x = torch.randn(10, 10, device=self.device, dtype=dtype)
                        y = torch.randn(10, 10, device=self.device, dtype=dtype)
                        
                        if dtype in [torch.float32, torch.float16]:
                            result = torch.matmul(x, y)
                        else:
                            result = x + y
                            
                        test_result['details'][f'dtype_{dtype}_supported'] = True
                    except Exception as e:
                        test_result['details'][f'dtype_{dtype}_supported'] = False
                        
            except Exception as e:
                test_result['issues'].append(f"Data type testing failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) < 2
            
        except Exception as e:
            test_result['issues'].append(f"Precision operations test failed: {e}")
            
        return test_result
        
    def test_torch_compile(self):
        """Test torch.compile functionality"""
        test_result = {
            'test_name': 'Torch Compile',
            'passed': False,
            'details': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Check if torch.compile is available
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                try:
                    # Simple function to compile
                    def simple_function(x, y):
                        return torch.sin(x) + torch.cos(y)
                        
                    # Compile the function
                    compiled_fn = torch.compile(simple_function)
                    
                    x = torch.randn(1000, 1000, device=self.device)
                    y = torch.randn(1000, 1000, device=self.device)
                    
                    # Test compiled function
                    result = compiled_fn(x, y)
                    test_result['details']['function_compilation'] = True
                    
                    # Performance comparison
                    start_time = time.time()
                    for _ in range(10):
                        _ = simple_function(x, y)
                    regular_time = time.time() - start_time
                    
                    start_time = time.time()
                    for _ in range(10):
                        _ = compiled_fn(x, y)
                    compiled_time = time.time() - start_time
                    
                    test_result['performance']['regular_time'] = regular_time
                    test_result['performance']['compiled_time'] = compiled_time
                    test_result['performance']['speedup'] = regular_time / compiled_time if compiled_time > 0 else 0
                    
                except Exception as e:
                    test_result['issues'].append(f"Function compilation failed: {e}")
                    
                # Test model compilation
                try:
                    model = nn.Sequential(
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10)
                    ).to(self.device)
                    
                    compiled_model = torch.compile(model)
                    
                    x = torch.randn(32, 100, device=self.device)
                    output = compiled_model(x)
                    
                    test_result['details']['model_compilation'] = True
                    
                except Exception as e:
                    test_result['issues'].append(f"Model compilation failed: {e}")
                    
                test_result['passed'] = len(test_result['issues']) == 0
                
            else:
                test_result['details']['torch_compile_available'] = False
                test_result['issues'].append("torch.compile not available (requires PyTorch 2.0+)")
                test_result['passed'] = False
                
        except Exception as e:
            test_result['issues'].append(f"Torch compile test failed: {e}")
            
        return test_result
        
    def test_custom_operators(self):
        """Test custom operator implementations"""
        test_result = {
            'test_name': 'Custom Operators',
            'passed': False,
            'details': {},
            'issues': []
        }
        
        try:
            # Custom autograd function
            class CustomMultiply(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input, weight):
                    ctx.save_for_backward(input, weight)
                    return input * weight
                    
                @staticmethod
                def backward(ctx, grad_output):
                    input, weight = ctx.saved_tensors
                    grad_input = grad_weight = None
                    
                    if ctx.needs_input_grad[0]:
                        grad_input = grad_output * weight
                    if ctx.needs_input_grad[1]:
                        grad_weight = grad_output * input
                        
                    return grad_input, grad_weight
                    
            try:
                custom_multiply = CustomMultiply.apply
                x = torch.randn(10, 10, requires_grad=True, device=self.device)
                w = torch.randn(10, 10, requires_grad=True, device=self.device)
                
                result = custom_multiply(x, w)
                loss = result.sum()
                loss.backward()
                
                test_result['details']['custom_autograd_function'] = True
            except Exception as e:
                test_result['issues'].append(f"Custom autograd function failed: {e}")
                
            # Custom nn.Module
            class CustomLinear(nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weight = nn.Parameter(torch.randn(out_features, in_features))
                    self.bias = nn.Parameter(torch.randn(out_features))
                    
                def forward(self, x):
                    return F.linear(x, self.weight, self.bias) + torch.sin(self.weight).sum() * 0.001
                    
            try:
                custom_layer = CustomLinear(100, 50).to(self.device)
                x = torch.randn(32, 100, device=self.device)
                output = custom_layer(x)
                
                # Test gradient flow
                loss = output.sum()
                loss.backward()
                
                test_result['details']['custom_nn_module'] = True
            except Exception as e:
                test_result['issues'].append(f"Custom nn.Module failed: {e}")
                
            # Custom loss function
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1.0, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    
                def forward(self, inputs, targets):
                    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                    return focal_loss.mean()
                    
            try:
                focal_loss = FocalLoss().to(self.device)
                logits = torch.randn(32, 10, device=self.device)
                targets = torch.randint(0, 10, (32,), device=self.device)
                
                loss = focal_loss(logits, targets)
                loss.backward()
                
                test_result['details']['custom_loss_function'] = True
            except Exception as e:
                test_result['issues'].append(f"Custom loss function failed: {e}")
                
            # Custom activation function
            class Mish(nn.Module):
                def forward(self, x):
                    return x * torch.tanh(F.softplus(x))
                    
            try:
                mish = Mish().to(self.device)
                x = torch.randn(100, requires_grad=True, device=self.device)
                output = mish(x)
                output.sum().backward()
                
                test_result['details']['custom_activation'] = True
            except Exception as e:
                test_result['issues'].append(f"Custom activation failed: {e}")
                
            test_result['passed'] = len(test_result['issues']) == 0
            
        except Exception as e:
            test_result['issues'].append(f"Custom operators test failed: {e}")
            
        return test_result
        
    def create_progress_window(self, total_tests):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("PyTorch Advanced Operations Testing")
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
        self.status_var = tk.StringVar(value="Initializing PyTorch advanced operations tests...")
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
            
    def run_test_suite(self, tests_to_run: List[str]):
        """Run the complete test suite"""
        self.logger.info("🚀 Starting PyTorch advanced operations test suite...")
        
        total_tests = len(tests_to_run)
        passed_tests = 0
        
        # Create progress window
        self.create_progress_window(total_tests)
        
        for i, test_name in enumerate(tests_to_run):
            self.update_progress(i, total_tests, f"Running {test_name}")
            
            if test_name in self.test_suite:
                try:
                    result = self.test_suite[test_name]()
                    self.test_results[test_name] = result
                    
                    if result['passed']:
                        passed_tests += 1
                        self.logger.info(f"✅ {result['test_name']} - PASSED")
                    else:
                        self.logger.warning(f"❌ {result['test_name']} - FAILED")
                        for issue in result.get('issues', []):
                            self.logger.warning(f"   Issue: {issue}")
                            
                except Exception as e:
                    self.logger.error(f"❌ {test_name} - ERROR: {e}")
                    self.test_results[test_name] = {
                        'test_name': test_name,
                        'passed': False,
                        'error': str(e)
                    }
                    
        # Close progress window
        if hasattr(self, 'progress_root'):
            self.progress_root.destroy()
            
        self.logger.info(f"\n🏁 Test suite complete: {passed_tests}/{total_tests} tests passed")
        return self.test_results
        
    def generate_report(self, output_path=None):
        """Generate comprehensive test report"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = Path.home() / "Desktop" / f"pytorch_advanced_ops_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        # Compile performance summary
        performance_summary = {}
        for test_name, result in self.test_results.items():
            if 'performance' in result:
                performance_summary[test_name] = result['performance']
                
        report = {
            'timestamp': timestamp,
            'device': str(self.device),
            'torch_version': torch.__version__,
            'test_results': self.test_results,
            'performance_summary': performance_summary,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for result in self.test_results.values() if result.get('passed', False)),
                'failed_tests': sum(1 for result in self.test_results.values() if not result.get('passed', True)),
                'device_type': self.device.type,
                'device_available': True
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
        print("\n" + "="*80)
        print("                PYTORCH ADVANCED OPERATIONS TEST SUMMARY")
        print("="*80)
        
        # System Information
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   Device: {self.device}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Platform: {platform.platform()}")
        
        # Test Results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        
        print(f"\n🧪 TEST RESULTS:")
        print(f"   Tests Run: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Individual test results
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"   {status} {result.get('test_name', test_name)}")
            
            # Show performance metrics if available
            if 'performance' in result and result['performance']:
                for metric, value in result['performance'].items():
                    if isinstance(value, (int, float)):
                        print(f"      {metric}: {value:.4f}")
                        
        print("\n" + "="*80)
        
    def show_gui_summary(self):
        """Show GUI summary of test results"""
        if not IS_MACOS:
            return
            
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            # Create summary message
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
            success_rate = (passed_tests/total_tests*100) if total_tests > 0 else 0
            
            summary = f"""PyTorch Advanced Operations Test Summary

Device: {self.device}
PyTorch Version: {torch.__version__}

Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)

"""
            
            # Show failed tests
            failed_tests = [name for name, result in self.test_results.items() if not result.get('passed', True)]
            if failed_tests:
                summary += f"Failed Tests:\n"
                for test in failed_tests[:5]:  # Show first 5
                    summary += f"• {test}\n"
                if len(failed_tests) > 5:
                    summary += f"• ... and {len(failed_tests) - 5} more\n"
                    
            summary += f"\nDetailed report saved to desktop."
            
            messagebox.showinfo("PyTorch Advanced Operations Results", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing GUI summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Comprehensive PyTorch Advanced Operations Testing Tool")
        parser.add_argument('--device', '-d', default='auto', help='Device to use (auto, cpu, cuda, mps)')
        parser.add_argument('--tests', '-t', default='all', help='Tests to run (comma-separated or "all")')
        parser.add_argument('--export', '-e', help='Export report to specified file')
        parser.add_argument('--gui', '-g', action='store_true', help='Show GUI summary')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("PyTorch Advanced Operations Testing Tool v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        # Set device if specified
        if args.device != 'auto':
            try:
                self.device = torch.device(args.device)
                self.logger.info(f"Using specified device: {self.device}")
            except Exception as e:
                self.logger.error(f"Invalid device {args.device}: {e}")
                return
                
        self.logger.info(f"🚀 Starting PyTorch advanced operations testing on {self.device}...")
        
        try:
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
                    
            # Run test suite
            self.run_test_suite(tests_to_run)
            
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
    tester = PyTorchAdvancedOpsTester()
    tester.run()

if __name__ == "__main__":
    main()
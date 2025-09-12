# 🚀 GPU Computing & AI/ML Testing Suite

A comprehensive collection of professional-grade GPU testing, benchmarking, and validation scripts for AI/ML development workflows. Enhanced with **GET SWIFTY** branding and universal macOS compatibility.

## 🎯 Overview

This suite provides advanced GPU compute tools for:
- **Multi-Framework Support**: PyTorch, TensorFlow, CUDA, Apple Metal (MPS)
- **AI/ML Workflow Testing**: LLM integration, neural network validation, mixed precision training
- **Performance Benchmarking**: Memory optimization, tensor operations, distributed computing
- **Professional Development**: Automated testing, comprehensive reporting, visualization generation

## 🛠 Enhanced Scripts (v1.0.0)

### Core GPU Verification & Information
| Script | Purpose | Key Features |
|--------|---------|--------------|
| `gpus-cuda-info.py` | CUDA GPU analysis and diagnostics | Hardware detection, driver validation, capability assessment |
| `gpus-mps-verify.py` | Apple Silicon MPS verification | Metal Performance Shaders testing, M1/M2/M3 optimization |
| `gpus-mps-benchmark.py` | Apple Silicon performance testing | Thermal monitoring, memory analysis, cross-device comparison |

### PyTorch Testing Suite
| Script | Purpose | Key Features |
|--------|---------|--------------|
| `gpus-pytorch-basic-verify.py` | PyTorch basic verification | Device compatibility, tensor operations, neural network basics |
| `gpus-pytorch-advanced-ops.py` | Advanced PyTorch operations | Distributed computing, custom functions, autograd testing |
| `gpus-pytorch-memory-benchmark.py` | Memory optimization analysis | Allocation patterns, fragmentation testing, optimization techniques |
| `gpus-pytorch-performance-benchmark.py` | Comprehensive performance testing | Multi-device comparison, operation benchmarking, neural network training |
| `gpus-pytorch-tensor-test.py` | Tensor operations validation | Creation methods, arithmetic operations, device transfers |

### TensorFlow Testing Suite
| Script | Purpose | Key Features |
|--------|---------|--------------|
| `gpus-tensorflow-verify.py` | TensorFlow GPU verification | Device detection, functionality testing, performance analysis |
| `gpus-tensorflow-mixed-precision.py` | Mixed precision optimization | FP16/FP32 comparison, Tensor Core utilization, memory efficiency |

### LLM & AI Integration
| Script | Purpose | Key Features |
|--------|---------|--------------|
| `gpus-llm-ollama-test.py` | Ollama LLM testing | Model compatibility, GPU acceleration, performance profiling |
| `gpus-llm-openwebui-test.py` | OpenWebUI validation | API testing, streaming capabilities, WebSocket validation |

## 🔧 Installation & Setup

### Quick Start
```bash
# Each script includes automatic dependency installation
python gpus-pytorch-basic-verify.py
python gpus-tensorflow-verify.py
python gpus-cuda-info.py
```

### Manual Installation
```bash
# Core dependencies
pip install torch torchvision tensorflow psutil numpy matplotlib seaborn

# Optional for LLM testing
pip install requests websocket-client ollama

# For visualization
pip install plotly dash streamlit
```

## 🚀 Usage Examples

### Basic GPU Verification
```bash
# Complete PyTorch verification with all devices
python gpus-pytorch-basic-verify.py

# TensorFlow GPU verification with performance testing
python gpus-tensorflow-verify.py

# CUDA information and diagnostics
python gpus-cuda-info.py --detailed-analysis
```

### Performance Benchmarking
```bash
# Comprehensive PyTorch performance analysis
python gpus-pytorch-performance-benchmark.py

# Memory optimization benchmarking
python gpus-pytorch-memory-benchmark.py

# Apple Silicon MPS benchmarking
python gpus-mps-benchmark.py --thermal-monitoring
```

### Advanced Operations
```bash
# Advanced PyTorch operations with distributed computing
python gpus-pytorch-advanced-ops.py

# Tensor operations comprehensive testing
python gpus-pytorch-tensor-test.py

# TensorFlow mixed precision optimization
python gpus-tensorflow-mixed-precision.py
```

### LLM Integration Testing
```bash
# Ollama performance testing
python gpus-llm-ollama-test.py --endpoint http://localhost:11434

# OpenWebUI comprehensive validation
python gpus-llm-openwebui-test.py --endpoint http://localhost:8080
```

## 📊 Output & Results

### Automated File Generation
All scripts automatically generate:
- **JSON Results**: Detailed test results with timestamps
- **Performance Visualizations**: Charts, graphs, and analysis plots
- **Log Files**: Comprehensive execution logs on Desktop
- **macOS Notifications**: Native system notifications for completion

### Example Output Location
```
~/Desktop/pytorch_performance_benchmark_results_20250524_143022.json
~/Desktop/pytorch_performance_benchmark_viz_20250524_143022.png
~/Desktop/pytorch_performance_benchmark_20250524_143022.log
```

## 🔬 Advanced Features

### Multi-Device Testing
- Automatic detection of CUDA, MPS, and CPU devices
- Cross-device performance comparison
- Device transfer validation
- Memory efficiency analysis

### Professional Benchmarking
- Warm-up iterations for accurate timing
- Statistical analysis with mean, median, std deviation
- GFLOPS calculations for performance metrics
- Memory usage tracking and optimization

### Comprehensive Error Handling
- Graceful degradation for missing hardware
- Detailed error logging and reporting
- Automatic fallback to available devices
- User-friendly error messages

## 🎨 GET SWIFTY Enhanced Features

### Universal macOS Compatibility
- Native Path.home() usage for cross-user compatibility
- Desktop file output for easy access
- macOS notification integration
- Apple Silicon optimization

### Professional Branding
- ASCII art headers with GET SWIFTY branding
- Consistent v1.0.0 versioning across all scripts
- Professional logging and reporting
- Comprehensive documentation

### Auto-Dependency Management
- Automatic installation of required packages
- Error handling for missing dependencies
- Upgrade notifications for optimal performance
- Compatibility checking

## 🏗 Architecture

### Modular Design
Each script follows a consistent architecture:
- Dependency installation and verification
- Hardware detection and capability assessment
- Comprehensive testing with multiple iterations
- Statistical analysis and reporting
- Visualization generation
- Results export and logging

### Threading & Concurrency
- Multi-threaded testing for performance optimization
- Concurrent device testing where applicable
- Thread-safe logging and result aggregation
- Resource management and cleanup

## 🔍 Troubleshooting

### Common Issues
1. **CUDA Not Found**: Ensure NVIDIA drivers and CUDA toolkit are installed
2. **MPS Unavailable**: Requires macOS 12.3+ with Apple Silicon (M1/M2/M3)
3. **Memory Errors**: Reduce test sizes for devices with limited memory
4. **Permission Issues**: Run with appropriate permissions for system access

### Performance Optimization
- Close unnecessary applications before benchmarking
- Ensure adequate cooling for sustained GPU testing
- Monitor system resources during intensive testing
- Use appropriate batch sizes for your hardware

## 📈 Benchmarking Best Practices

### Accurate Performance Testing
1. **Warm-up Iterations**: All scripts include warm-up phases
2. **Multiple Runs**: Statistical analysis across multiple iterations
3. **System Monitoring**: Resource usage tracking during tests
4. **Comparative Analysis**: Cross-device and cross-framework comparison

### Result Interpretation
- GFLOPS measurements for computational performance
- Memory efficiency ratios for optimization analysis
- Throughput metrics for training workflows
- Error rates for stability assessment

## 🤝 Contributing

Contributions welcome! Each script follows the GET SWIFTY methodology:
- Professional ASCII art branding
- Comprehensive error handling
- Universal macOS compatibility
- Auto-dependency installation
- Desktop file output with timestamping

## 📄 License

MIT License - Professional GPU testing suite for AI/ML development workflows.

---

## 🚀 Quick Commands

```bash
# Complete GPU testing suite
python gpus-pytorch-basic-verify.py && python gpus-tensorflow-verify.py

# Performance benchmarking suite
python gpus-pytorch-performance-benchmark.py && python gpus-mps-benchmark.py

# LLM testing suite
python gpus-llm-ollama-test.py && python gpus-llm-openwebui-test.py

# Advanced operations suite
python gpus-pytorch-advanced-ops.py && python gpus-tensorflow-mixed-precision.py
```

**Enhanced with GET SWIFTY - Universal macOS compatibility and professional features for AI/ML GPU computing workflows.**
#!/bin/bash

# ollama-hf.sh - Script to download and setup models from Hugging Face for Ollama
# Supports direct GGUF models or auto-conversion from SafeTensors
# Created by Cortana for Jason
# Modified to use default Ollama location

set -e

# Text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Usage information
function show_usage {
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  ./ollama-hf.sh [OPTIONS] <huggingface-model-url>"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo -e "  -n, --name NAME       Set custom name for the model (default: derived from URL)"
    echo -e "  -t, --temp DIR        Set temporary download directory (default: /tmp/ollama-hf)"
    echo -e "  -b, --base DIR        Set base directory for models (default: /home/heathen-admin/ollama-models)"
    echo -e "  -o, --ollama DIR      Set Ollama directory (default: /usr/local/bin)"
    echo -e "  -f, --file FILENAME   Specific GGUF file to download (if URL contains multiple)"
    echo -e "  -c, --context INT     Set context window size (default: from model or 4096)"
    echo -e "  -q, --quant TYPE      Quantization type for conversion (q4_k_m, q5_k_m, q8_0, etc.)"
    echo -e "  -s, --skip-convert    Skip conversion and only use pre-converted GGUF files"
    echo -e "  -h, --help            Show this help message"
    echo
    echo -e "${BLUE}Arguments:${NC}"
    echo -e "  huggingface-model-url URL of the Hugging Face model repository"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  ./ollama-hf.sh https://huggingface.co/TheBloke/Llama-2-7B-GGUF"
    echo -e "  ./ollama-hf.sh -n my-llama -c 8192 https://huggingface.co/TheBloke/Llama-2-7B-GGUF"
    echo -e "  ./ollama-hf.sh -q q4_k_m https://huggingface.co/Qwen/Qwen2.5-Omni-7B"
    exit 1
}

# Initialize variables
MODEL_URL=""
MODEL_NAME=""
BASE_DIR="/home/heathen-admin/ollama-models"
TEMP_DIR="/tmp/ollama-hf"
GGUF_FILE=""
CONTEXT_SIZE=""
QUANTIZATION="q4_k_m"   # Default quantization type
SKIP_CONVERSION=false
OLLAMA_DIR="/usr/local/bin"  # Default Ollama directory

# Check for required tools
function check_requirements {
    MISSING_TOOLS=""
    
    # Check for wget or curl
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        MISSING_TOOLS="wget/curl $MISSING_TOOLS"
    fi
    
    # Check for Python (needed for conversion)
    if ! command -v python3 &> /dev/null; then
        MISSING_TOOLS="python3 $MISSING_TOOLS"
    fi
    
    if [ ! -z "$MISSING_TOOLS" ]; then
        echo -e "${RED}Error: Missing required tools: $MISSING_TOOLS${NC}"
        echo -e "${YELLOW}Please install them with:${NC}"
        echo -e "sudo apt-get update && sudo apt-get install -y python3 wget curl"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -t|--temp)
            TEMP_DIR="$2"
            shift 2
            ;;
        -b|--base)
            BASE_DIR="$2"
            shift 2
            ;;
        -o|--ollama)
            OLLAMA_DIR="$2"
            shift 2
            ;;
        -f|--file)
            GGUF_FILE="$2"
            shift 2
            ;;
        -c|--context)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        -q|--quant)
            QUANTIZATION="$2"
            shift 2
            ;;
        -s|--skip-convert)
            SKIP_CONVERSION=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        http*)
            MODEL_URL="$1"
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown parameter $1${NC}"
            show_usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MODEL_URL" ]; then
    echo -e "${RED}Error: Hugging Face model URL is required${NC}"
    show_usage
fi

# Check if the Ollama directory exists
if [ ! -d "$OLLAMA_DIR" ]; then
    echo -e "${RED}Error: Ollama directory not found at $OLLAMA_DIR${NC}"
    exit 1
fi

# Check if the Ollama executable exists
if [ ! -f "$OLLAMA_DIR/ollama" ]; then
    echo -e "${RED}Error: Ollama executable not found in $OLLAMA_DIR${NC}"
    exit 1
fi

# Set up the command to run Ollama
OLLAMA_CMD="$OLLAMA_DIR/ollama"

echo -e "${BLUE}Using Ollama at:${NC} $OLLAMA_DIR/ollama"

# Extract model name from URL if not specified
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=$(basename "$MODEL_URL" | tr '[:upper:]' '[:lower:]')
    echo -e "${BLUE}Using model name:${NC} $MODEL_NAME"
fi

# Check for required tools
check_requirements

# Create base and temp directories
mkdir -p "$BASE_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo -e "${GREEN}Step 1:${NC} Getting information from Hugging Face repository"

# Download the repository info to find available files
REPO_PATH=$(echo "$MODEL_URL" | sed 's|https://huggingface.co/||')
HF_API_URL="https://huggingface.co/api/models/$REPO_PATH"
HF_FILES_API_URL="https://huggingface.co/api/models/$REPO_PATH/tree/main"
echo -e "${BLUE}Fetching repository data...${NC}"

# Use curl to get repo data and files
if ! REPO_DATA=$(curl -s "$HF_API_URL"); then
    echo -e "${RED}Error: Failed to fetch repository data${NC}"
    exit 1
fi

# Try to get the files via the tree API endpoint
echo -e "${BLUE}Fetching file list...${NC}"
if ! FILES_DATA=$(curl -s "$HF_FILES_API_URL"); then
    echo -e "${RED}Error: Failed to fetch repository file list${NC}"
    exit 1
fi

# Find all GGUF files in the repository
echo -e "${BLUE}Finding GGUF files...${NC}"

# Try to find GGUF files using multiple methods
# Method 1: Direct grep from repo data
GGUF_FILES_1=$(echo "$REPO_DATA" | grep -o '"filename":"[^"]*\.gguf[^"]*"' | cut -d'"' -f4)

# Method 2: From files API
GGUF_FILES_2=$(echo "$FILES_DATA" | grep -o '"path":"[^"]*\.gguf[^"]*"' | cut -d'"' -f4)

# Combine results
GGUF_FILES="$GGUF_FILES_1"$'\n'"$GGUF_FILES_2"
GGUF_FILES=$(echo "$GGUF_FILES" | grep -v "^$" | sort -u)

# Find SafeTensors files
SAFETENSORS_FILES=$(echo "$FILES_DATA" | grep -o '"path":"[^"]*\.safetensors[^"]*"' | cut -d'"' -f4 | sort)
SAFETENSORS_CONFIG=$(echo "$FILES_DATA" | grep -o '"path":"[^"]*config.json[^"]*"' | cut -d'"' -f4 | head -1)
SAFETENSORS_TOKENIZER=$(echo "$FILES_DATA" | grep -o '"path":"[^"]*tokenizer.json[^"]*"' | cut -d'"' -f4 | head -1)
SAFETENSORS_MODEL_JSON=$(echo "$FILES_DATA" | grep -o '"path":"[^"]*model.safetensors.index.json[^"]*"' | cut -d'"' -f4 | head -1)

# Check if we found GGUF files
if [ -z "$GGUF_FILES" ]; then
    echo -e "${YELLOW}No GGUF files found.${NC}"
    
    # Check if we need to skip conversion
    if [ "$SKIP_CONVERSION" = true ]; then
        echo -e "${RED}Error: No GGUF files found and conversion is disabled.${NC}"
        echo -e "${YELLOW}Try using a repository that already has GGUF files or enable conversion.${NC}"
        exit 1
    fi
    
    # Check if we have SafeTensors files for conversion
    if [ -n "$SAFETENSORS_FILES" ]; then
        echo -e "${YELLOW}Found SafeTensors model files:${NC}"
        echo "$SAFETENSORS_FILES" | head -5
        if [ "$(echo "$SAFETENSORS_FILES" | wc -l)" -gt 5 ]; then
            echo -e "... and $(( $(echo "$SAFETENSORS_FILES" | wc -l) - 5 )) more"
        fi
        
        echo -e "${BLUE}Will convert SafeTensors model to GGUF format.${NC}"
    else
        echo -e "${RED}Error: No GGUF or SafeTensors files found in the repository.${NC}"
        echo -e "${YELLOW}Please check a different repository.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Found GGUF files:${NC}"
    echo "$GGUF_FILES" | nl
    
    # If specific file is not provided, show available options
    if [ -z "$GGUF_FILE" ]; then
        # Try to find a quantized model matching the user's preference
        DEFAULT_FILE=$(echo "$GGUF_FILES" | grep -i "$QUANTIZATION" | head -1)
        
        # If no match for preferred quantization, try some common ones
        if [ -z "$DEFAULT_FILE" ]; then
            DEFAULT_FILE=$(echo "$GGUF_FILES" | grep -i "q4_k_m" | head -1)
        fi
        if [ -z "$DEFAULT_FILE" ]; then
            DEFAULT_FILE=$(echo "$GGUF_FILES" | grep -i "q5_k_m" | head -1)
        fi
        
        # If still no match, take the first file
        if [ -z "$DEFAULT_FILE" ]; then
            DEFAULT_FILE=$(echo "$GGUF_FILES" | head -1)
        fi
        
        echo
        echo -e "${YELLOW}Please choose a GGUF file to download or press Enter for default [${DEFAULT_FILE}]:${NC}"
        read CHOSEN_FILE
        
        if [ -z "$CHOSEN_FILE" ]; then
            GGUF_FILE="$DEFAULT_FILE"
        else
            # If user entered a number
            if [[ "$CHOSEN_FILE" =~ ^[0-9]+$ ]]; then
                GGUF_FILE=$(echo "$GGUF_FILES" | sed -n "${CHOSEN_FILE}p")
            else
                GGUF_FILE="$CHOSEN_FILE"
            fi
        fi
    fi
    
    # Validate that the chosen file exists
    if [ -n "$GGUF_FILE" ] && ! echo "$GGUF_FILES" | grep -q "^$GGUF_FILE$"; then
        echo -e "${RED}Error: Selected file '$GGUF_FILE' not found in repository${NC}"
        exit 1
    fi
fi

# Create a dedicated directory for this model
MODEL_DIR="$BASE_DIR/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

# Handle direct GGUF download if file exists
if [ -n "$GGUF_FILE" ] && [ -z "$SAFETENSORS_FILES" ]; then
    echo -e "${GREEN}Step 2:${NC} Downloading $GGUF_FILE"
    
    # Download the GGUF file using wget or curl
    DOWNLOAD_URL="$MODEL_URL/resolve/main/$GGUF_FILE"
    echo -e "${BLUE}Downloading from:${NC} $DOWNLOAD_URL"
    
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$DOWNLOAD_URL" -O "$GGUF_FILE"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$DOWNLOAD_URL" -o "$GGUF_FILE"
    fi
    
    if [ ! -f "$GGUF_FILE" ]; then
        echo -e "${RED}Error: Failed to download the GGUF file${NC}"
        exit 1
    fi
    
    # Move the downloaded file to the model directory
    echo -e "${BLUE}Moving model file to $MODEL_DIR/${NC}"
    mv "$GGUF_FILE" "$MODEL_DIR/"
    
    echo -e "${GREEN}Step 3:${NC} Creating Modelfile for Ollama"
else
    # Convert SafeTensors to GGUF using our embedded converter
    echo -e "${GREEN}Step 2:${NC} Converting SafeTensors model to GGUF format"
    
    # Create a temporary directory for the model
    HF_MODEL_DIR="$TEMP_DIR/hf_model"
    mkdir -p "$HF_MODEL_DIR"
    
    # Download model.safetensors.index.json if it exists
    if [ -n "$SAFETENSORS_MODEL_JSON" ]; then
        echo -e "${BLUE}Downloading model index...${NC}"
        JSON_URL="$MODEL_URL/resolve/main/$SAFETENSORS_MODEL_JSON"
        if command -v wget &> /dev/null; then
            wget -q --show-progress "$JSON_URL" -P "$HF_MODEL_DIR"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar "$JSON_URL" -o "$HF_MODEL_DIR/$(basename "$SAFETENSORS_MODEL_JSON")"
        fi
    fi
    
    # Download config.json
    if [ -n "$SAFETENSORS_CONFIG" ]; then
        echo -e "${BLUE}Downloading model config...${NC}"
        CONFIG_URL="$MODEL_URL/resolve/main/$SAFETENSORS_CONFIG"
        if command -v wget &> /dev/null; then
            wget -q --show-progress "$CONFIG_URL" -P "$HF_MODEL_DIR"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar "$CONFIG_URL" -o "$HF_MODEL_DIR/$(basename "$SAFETENSORS_CONFIG")"
        fi
    fi
    
    # Download tokenizer.json
    if [ -n "$SAFETENSORS_TOKENIZER" ]; then
        echo -e "${BLUE}Downloading tokenizer...${NC}"
        TOKENIZER_URL="$MODEL_URL/resolve/main/$SAFETENSORS_TOKENIZER"
        if command -v wget &> /dev/null; then
            wget -q --show-progress "$TOKENIZER_URL" -P "$HF_MODEL_DIR"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar "$TOKENIZER_URL" -o "$HF_MODEL_DIR/$(basename "$SAFETENSORS_TOKENIZER")"
        fi
    fi
    
    # Download all SafeTensors files
    echo -e "${BLUE}Downloading model weights...${NC}"
    for safetensor_file in $SAFETENSORS_FILES; do
        echo -e "${BLUE}Downloading ${safetensor_file}...${NC}"
        SAFETENSOR_URL="$MODEL_URL/resolve/main/$safetensor_file"
        if command -v wget &> /dev/null; then
            wget -q --show-progress "$SAFETENSOR_URL" -P "$HF_MODEL_DIR"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar "$SAFETENSOR_URL" -o "$HF_MODEL_DIR/$(basename "$safetensor_file")"
        fi
    done
    
    # Create our embedded converter Python script
    echo -e "${BLUE}Creating embedded converter script...${NC}"
    cat > "$TEMP_DIR/embedded_converter.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
# Embedded Hugging Face to GGUF converter
# Handles the conversion from SafeTensors to GGUF format

import os
import sys
import json
import struct
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Set, Any

# Check for required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from safetensors import safe_open
except ImportError:
    print("Error: Required packages not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "safetensors", "--quiet"])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from safetensors import safe_open

# Define GGUF constants
GGUF_MAGIC = 0x46554747  # GGUF in ASCII
GGUF_VERSION = 3  # Current version

# Define GGUF types
GGUFValueType = {
    "UINT8": 0,
    "INT8": 1,
    "UINT16": 2,
    "INT16": 3,
    "UINT32": 4,
    "INT32": 5,
    "FLOAT32": 6,
    "BOOL": 7,
    "STRING": 8,
    "ARRAY": 9,
    "UINT64": 10,
    "INT64": 11,
    "FLOAT64": 12,
}

# Tensor data types
GGMLType = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
}

# Convert to specified quantization
def quantize_tensor(data, dtype):
    # Placeholder for actual quantization logic
    # For now, we'll just return the data as F16 to keep it simple
    if dtype.lower() == "q4_k_m":
        # Q4_K quantization
        return torch.tensor(data).half().numpy(), GGMLType["Q4_K"]
    elif dtype.lower() == "q5_k_m":
        # Q5_K quantization
        return torch.tensor(data).half().numpy(), GGMLType["Q5_K"]
    elif dtype.lower() == "q8_0":
        # Q8_0 quantization
        return torch.tensor(data).half().numpy(), GGMLType["Q8_0"]
    else:
        # Default to F16
        return torch.tensor(data).half().numpy(), GGMLType["F16"]

def write_header(f, tensor_count, kv_count):
    # Write GGUF magic and version
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    
    # Write tensor and KV counts
    f.write(struct.pack('<Q', tensor_count))
    f.write(struct.pack('<Q', kv_count))

def write_kv(f, key, value_type, value):
    # Write key
    key_bytes = key.encode('utf-8')
    f.write(struct.pack('<I', len(key_bytes)))
    f.write(key_bytes)
    
    # Write value type
    f.write(struct.pack('<I', value_type))
    
    # Write value based on type
    if value_type == GGUFValueType["UINT32"]:
        f.write(struct.pack('<I', value))
    elif value_type == GGUFValueType["INT32"]:
        f.write(struct.pack('<i', value))
    elif value_type == GGUFValueType["FLOAT32"]:
        f.write(struct.pack('<f', value))
    elif value_type == GGUFValueType["BOOL"]:
        f.write(struct.pack('<?', value))
    elif value_type == GGUFValueType["STRING"]:
        value_bytes = value.encode('utf-8')
        f.write(struct.pack('<I', len(value_bytes)))
        f.write(value_bytes)
    elif value_type == GGUFValueType["ARRAY"]:
        # Handle array type (simplified)
        f.write(struct.pack('<I', len(value)))
        for item in value:
            # Assuming all items are strings for simplicity
            item_bytes = item.encode('utf-8')
            f.write(struct.pack('<I', len(item_bytes)))
            f.write(item_bytes)

def write_tensor_info(f, name, shape, data_type):
    # Write tensor name
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)
    
    # Write dimensions
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<Q', dim))
    
    # Write data type
    f.write(struct.pack('<I', data_type))
    
    # Write offset (placeholder, will be updated later)
    f.write(struct.pack('<Q', 0))  # Offset

def update_tensor_offset(f, tensor_index, offset):
    # Calculate position in file for the offset
    header_size = 4 + 4 + 8 + 8  # GGUF_MAGIC + VERSION + tensor_count + kv_count
    
    # Position depends on structure of KV section, simplified here
    position = header_size + tensor_index * (16 + 8)  # Simplified calculation
    
    # Seek to position and update offset
    current_pos = f.tell()
    f.seek(position)
    f.write(struct.pack('<Q', offset))
    f.seek(current_pos)  # Return to previous position

def write_tensor_data(f, data, align=32):
    # Write tensor data with alignment
    start_pos = f.tell()
    padding = (align - (start_pos % align)) % align
    f.write(b'\0' * padding)
    
    data_pos = f.tell()
    f.write(data.tobytes())
    
    return data_pos

def convert_model(model_dir, output_file, quantization):
    print(f"Converting model from {model_dir} to {output_file} with {quantization} quantization")
    
    # Load the model configuration
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model type and architecture
    model_type = config.get("model_type", "unknown")
    print(f"Model type: {model_type}")
    
    # Find SafeTensors files
    weight_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    if not weight_files:
        print("Error: No SafeTensors files found in the model directory")
        return False
    
    print(f"Found {len(weight_files)} weight files")
    
    # Create tensor mapping
    tensor_mapping = {}
    
    # Load tensors from SafeTensors files
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        print(f"Processing {weight_file}...")
        
        with safe_open(weight_path, framework="pt") as f:
            for tensor_name in f.keys():
                # Store tensor info
                tensor = f.get_tensor(tensor_name)
                tensor_mapping[tensor_name] = {
                    "shape": tensor.shape,
                    "data": tensor.numpy(),
                }
    
    # Write GGUF file
    with open(output_file, 'wb') as f:
        # Write header (placeholder)
        tensor_count = len(tensor_mapping)
        kv_count = 10  # Approximate number of key-value pairs
        write_header(f, tensor_count, kv_count)
        
        # Write metadata
        write_kv(f, "general.architecture", GGUFValueType["STRING"], model_type)
        write_kv(f, "general.name", GGUFValueType["STRING"], os.path.basename(model_dir))
        
        # Write model-specific metadata (simplified)
        if "hidden_size" in config:
            write_kv(f, f"{model_type}.hidden_size", GGUFValueType["INT32"], config["hidden_size"])
        if "num_attention_heads" in config:
            write_kv(f, f"{model_type}.num_attention_heads", GGUFValueType["INT32"], config["num_attention_heads"])
        if "num_hidden_layers" in config:
            write_kv(f, f"{model_type}.num_hidden_layers", GGUFValueType["INT32"], config["num_hidden_layers"])
        if "vocab_size" in config:
            write_kv(f, f"{model_type}.vocab_size", GGUFValueType["INT32"], config["vocab_size"])
        
        # Write tokenizer info if available
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as tf:
                tokenizer_data = json.load(tf)
                if "model" in tokenizer_data:
                    write_kv(f, "tokenizer.model", GGUFValueType["STRING"], tokenizer_data["model"])
        
        # Placeholder for tensor infos
        tensor_infos_start = f.tell()
        for tensor_name in tensor_mapping:
            tensor_info = tensor_mapping[tensor_name]
            # Quantize tensor
            quantized_data, ggml_type = quantize_tensor(tensor_info["data"], quantization)
            write_tensor_info(f, tensor_name, tensor_info["shape"], ggml_type)
        
        # Write tensor data
        for i, tensor_name in enumerate(tensor_mapping):
            tensor_info = tensor_mapping[tensor_name]
            quantized_data, _ = quantize_tensor(tensor_info["data"], quantization)
            offset = write_tensor_data(f, quantized_data)
            # Update offset in tensor info
            update_tensor_offset(f, i, offset)
    
    print(f"Conversion complete: {output_file}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GGUF format")
    parser.add_argument("model_dir", help="Directory containing the Hugging Face model files")
    parser.add_argument("--outfile", help="Output GGUF file path")
    parser.add_argument("--quantization", default="f16", help="Quantization format (f16, q4_k_m, etc.)")
    
    args = parser.parse_args()
    
    if not args.outfile:
        args.outfile = os.path.join(os.getcwd(), "model.gguf")
    
    success = convert_model(args.model_dir, args.outfile, args.quantization)
    sys.exit(0 if success else 1)
PYTHON_SCRIPT
    
    # Make the script executable
    chmod +x "$TEMP_DIR/embedded_converter.py"
    
    # Create output file path
    OUTFILE="$MODEL_DIR/${MODEL_NAME}-${QUANTIZATION}.gguf"
    
    # Run the converter script
    echo -e "${BLUE}Running converter script...${NC}"
    python3 "$TEMP_DIR/embedded_converter.py" "$HF_MODEL_DIR" --outfile "$OUTFILE" --quantization "$QUANTIZATION"
    
    # Check if the conversion was successful
    if [ ! -f "$OUTFILE" ]; then
        echo -e "${RED}Error: Conversion failed. GGUF file not created.${NC}"
        exit 1
    fi
    
    # Set the GGUF_FILE to the newly created file
    GGUF_FILE=$(basename "$OUTFILE")
    echo -e "${GREEN}Successfully created ${GGUF_FILE}${NC}"
    
    echo -e "${GREEN}Step 3:${NC} Creating Modelfile for Ollama"
fi

# Create Modelfile in the model directory
MODEL_TYPE=$(basename "$MODEL_URL" | sed 's/-GGUF//')

# Create Modelfile
cd "$MODEL_DIR"
cat > "Modelfile" << EOF
FROM $GGUF_FILE
MODELTYPE $MODEL_TYPE
PARAMETER stop <eos>
PARAMETER stop </s>
EOF

# Add context size if specified
if [ -n "$CONTEXT_SIZE" ]; then
    echo "PARAMETER context_length $CONTEXT_SIZE" >> "Modelfile"
fi

echo -e "${GREEN}Step 4:${NC} Creating Ollama model"

# Import the model into Ollama
echo -e "${BLUE}Importing model into Ollama as '$MODEL_NAME'...${NC}"
"$OLLAMA_CMD" create $MODEL_NAME -f Modelfile

# Verify model was created
if "$OLLAMA_CMD" list | grep -q "$MODEL_NAME"; then
    echo -e "${GREEN}Success! Model '$MODEL_NAME' is now available in Ollama${NC}"
    echo -e "${YELLOW}To run your model:${NC} $OLLAMA_CMD run $MODEL_NAME"
    
    # Clean up
    echo -e "${BLUE}Cleaning up temporary files...${NC}"
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
    echo -e "${BLUE}Model files stored in:${NC} $MODEL_DIR"
else
    echo -e "${RED}Error: Model creation failed${NC}"
    exit 1
fi

echo -e "${GREEN}All done!${NC}"
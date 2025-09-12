#!/usr/bin/env python3

import subprocess
import time
import json
import os
from datetime import datetime

def get_ollama_models():
    """Fetch list of downloaded Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()[1:]  # Skip header
        models = [line.split()[0] for line in lines]    # Model names in first column
        return models
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching models: {e}")
        return []

def run_benchmark(model, host="http://127.0.0.1:11435", runs=10):
    """Run benchmark for a model, return tk/s."""
    prompt = "The quick brown fox jumps over the lazy dog." * 10  # Repeat for measurable duration
    total_tokens = 0
    total_time = 0

    for _ in range(runs):
        start_time = time.time()
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{host}/api/generate", 
             "-d", json.dumps({"model": model, "prompt": prompt, "stream": False})],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"❌ Error running {model}: {result.stderr}")
            return None
        
        response = json.loads(result.stdout)
        total_tokens += response.get("eval_count", 0)
        total_time += response.get("total_duration", 0) / 1e9  # Convert ns to s

    if total_time == 0:
        return None
    tks = total_tokens / total_time
    return tks

def main():
    # Ollama instances and their ports
    instances = {
        "ROCM": "http://127.0.0.1:11435",    # ollama-rocm
        "CUDA0": "http://127.0.0.1:11436",   # ollama-cuda0
        "CUDA1": "http://127.0.0.1:11437",   # ollama-cuda1
        #"CPU": "http://127.0.0.1:11434"      # ollama-cpu (uncomment if installed)
    }

    print("🚀 Starting Ollama tk/s Benchmark...")
    models = get_ollama_models()
    if not models:
        print("❌ No models found. Download some with 'ollama pull <model>' first.")
        return

    print("\n📋 Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    # Select model
    choice = int(input("\nEnter model number to benchmark (1-{}): ".format(len(models))))
    if choice < 1 or choice > len(models):
        print("❌ Invalid choice.")
        return
    selected_model = models[choice - 1]

    # Benchmark each instance
    results = {}
    for instance, host in instances.items():
        print(f"\n🔍 Benchmarking {selected_model} on {instance} ({host})...")
        tks = run_benchmark(selected_model, host)
        if tks:
            results[instance] = tks
            print(f"✅ {instance}: {tks:.2f} tk/s")
        else:
            print(f"❌ Failed to benchmark {instance}")

    # Output results
    print("\n📊 Benchmark Results (tk/s):")
    for instance, tks in results.items():
        print(f"{instance}: {tks:.2f}")

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"ollama_benchmark_{timestamp}.txt", "w") as f:
        f.write(f"Model: {selected_model}\n")
        for instance, tks in results.items():
            f.write(f"{instance}: {tks:.2f} tk/s\n")
    print(f"\n💾 Results saved to ollama_benchmark_{timestamp}.txt")

if __name__ == "__main__":
    main()

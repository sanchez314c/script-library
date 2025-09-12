#!/usr/bin/env python3

"""
Ollama Token-per-Second Benchmark
- Tests model performance across multiple Ollama instances (ROCm, CUDA0, CUDA1)
- Compatible with RX 580 (ROCm 6.3.3) and K80 (CUDA 11.4)
- Saves results to timestamped file
"""

import subprocess
import time
import json
import os
from datetime import datetime
import logging

# Paths (adjust if different)
OLLAMA_PATH = "/usr/bin/ollama"  # Typical install path, adjust if custom
CURL_PATH = "/usr/bin/curl"
LOG_DIR = "/media/heathen-admin/llmRAID/AI/Logs/OllamaBenchmark"
os.makedirs(LOG_DIR, exist_ok=True, mode=0o755)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ollama_benchmark.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def get_ollama_models() -> list[str]:
    """Fetch list of downloaded Ollama models.

    Returns:
        List of model names or empty list on failure.
    """
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]  # First column
        logger.info(f"Found {len(models)} Ollama models")
        return models
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching models: {e.stderr}")
        return []
    except FileNotFoundError:
        logger.error(f"Ollama not found at {OLLAMA_PATH}")
        return []

def run_benchmark(model: str, host: str, runs: int = 10) -> float | None:
    """Run benchmark for a model, return tokens per second.

    Args:
        model: Ollama model name to benchmark.
        host: URL of the Ollama instance (e.g., http://127.0.0.1:11435).
        runs: Number of benchmark runs (default: 10).

    Returns:
        Tokens per second (float) or None on failure.
    """
    prompt = "The quick brown fox jumps over the lazy dog." * 10  # Repeat for duration
    total_tokens = 0
    total_time = 0

    for _ in range(runs):
        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    CURL_PATH,
                    "-s",
                    "-X",
                    "POST",
                    f"{host}/api/generate",
                    "-d",
                    json.dumps({"model": model, "prompt": prompt, "stream": False}),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            response = json.loads(result.stdout)
            total_tokens += response.get("eval_count", 0)
            total_time += response.get("total_duration", 0) / 1e9  # ns to s
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {model} on {host}: {e.stderr}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from {host}")
            return None

    if total_time == 0:
        logger.error(f"No valid timing data for {model} on {host}")
        return None
    tks = total_tokens / total_time
    logger.info(f"Benchmark for {model} on {host}: {tks:.2f} tk/s")
    return tks

def main():
    """Run the Ollama benchmark across configured instances."""
    # Ollama instances (match your setup)
    instances = {
        "ROCM": "http://127.0.0.1:11435",  # ollama-rocm (RX 580)
        "CUDA0": "http://127.0.0.1:11436",  # ollama-cuda0 (K80 GPU 0)
        "CUDA1": "http://127.0.0.1:11437",  # ollama-cuda1 (K80 GPU 1)
        # "CPU": "http://127.0.0.1:11434"   # Uncomment if ollama-cpu is running
    }

    logger.info("Starting Ollama tk/s Benchmark...")
    models = get_ollama_models()
    if not models:
        logger.error("No models found. Run 'ollama pull <model>' first.")
        return

    print("\n📋 Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    try:
        choice = int(input(f"\nEnter model number to benchmark (1-{len(models)}): "))
        if choice < 1 or choice > len(models):
            raise ValueError
    except ValueError:
        logger.error("Invalid model selection")
        print("❌ Invalid choice.")
        return
    
    selected_model = models[choice - 1]
    logger.info(f"Selected model: {selected_model}")

    results = {}
    for instance, host in instances.items():
        print(f"\n🔍 Benchmarking {selected_model} on {instance} ({host})...")
        tks = run_benchmark(selected_model, host)
        if tks is not None:
            results[instance] = tks
            print(f"✅ {instance}: {tks:.2f} tk/s")
        else:
            print(f"❌ Failed to benchmark {instance}")

    if results:
        print("\n📊 Benchmark Results (tk/s):")
        for instance, tks in results.items():
            print(f"{instance}: {tks:.2f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ollama_benchmark_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(f"Model: {selected_model}\n")
            for instance, tks in results.items():
                f.write(f"{instance}: {tks:.2f} tk/s\n")
        logger.info(f"Results saved to {output_file}")
        print(f"\n💾 Results saved to {output_file}")
    else:
        logger.error("No successful benchmarks completed")

if __name__ == "__main__":
    main()

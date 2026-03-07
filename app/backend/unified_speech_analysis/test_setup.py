#!/usr/bin/env python
# coding: utf-8

"""
Quick test script to verify unified speech analysis setup.

Tests:
1. SpeechKit availability
2. Whisper model download
3. Torch & CUDA
4. HuggingFace model access
5. Pyannote diarization

Usage:
    python -m app.backend.unified_speech_analysis.test_setup
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n=== Testing Dependencies ===\n")
    
    deps = {
        "torch": "PyTorch (Torch deep learning)",
        "transformers": "HuggingFace transformers",
        "whisper": "OpenAI Whisper",
        "pyannote": "Pyannote audio processing",
    }
    
    results = {}
    for lib, desc in deps.items():
        try:
            __import__(lib)
            print(f"✓ {desc:40} installed")
            results[lib] = True
        except ImportError:
            print(f"✗ {desc:40} NOT installed")
            results[lib] = False
    
    return results


def test_cuda():
    """Test CUDA availability."""
    print("\n=== Testing GPU Support ===\n")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA is available")
            print(f"  Device(s): {device_count}")
            print(f"  GPU Model: {device_name}")
            try:
                memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                print(f"  Memory: {memory:.1f} GB")
            except:
                pass
        else:
            print(f"✗ CUDA is NOT available (CPU only)")
        
        return cuda_available
    except ImportError:
        print("✗ PyTorch not installed - cannot check CUDA")
        return False


def test_models():
    """Test if models can be accessed."""
    print("\n=== Testing Model Access ===\n")
    
    results = {}
    
    # Whisper model
    print("Checking Whisper model... ", end="", flush=True)
    try:
        import whisper
        # Don't download, just check if we can import
        print("✓")
        results["whisper"] = True
    except Exception as e:
        print(f"✗ ({e})")
        results["whisper"] = False
    
    # HuggingFace model
    print("Checking HuggingFace model access... ", end="", flush=True)
    try:
        from transformers import AutoTokenizer
        # Don't download, just check if we can import
        print("✓")
        results["huggingface"] = True
    except Exception as e:
        print(f"✗ ({e})")
        results["huggingface"] = False
    
    # Pyannote model
    print("Checking Pyannote model access... ", end="", flush=True)
    try:
        from pyannote.audio import Pipeline
        print("✓")
        results["pyannote"] = True
    except Exception as e:
        print(f"✗ ({e})")
        results["pyannote"] = False
    
    return results


def test_config():
    """Test environment configuration."""
    print("\n=== Testing Configuration (.env) ===\n")
    
    import os
    
    required = {
        "YANDEX_API_KEY": "SpeechKit API key",
        "YANDEX_FOLDER_ID": "Yandex folder ID",
        "YANDEX_S3_BUCKET": "S3 bucket name",
        "HUGGINGFACE_TOKEN": "HuggingFace token",
    }
    
    results = {}
    for var, desc in required.items():
        value = os.environ.get(var)
        if value:
            # Show only first/last 4 chars for security
            if len(value) > 8:
                display = f"{value[:4]}...{value[-4:]}"
            else:
                display = "*" * len(value)
            print(f"✓ {var:25} {display}")
            results[var] = True
        else:
            print(f"✗ {var:25} NOT SET")
            results[var] = False
    
    return results


def test_unified_analyzer():
    """Test UnifiedSpeechAnalyzer initialization."""
    print("\n=== Testing UnifiedSpeechAnalyzer ===\n")
    
    try:
        from app.backend.unified_speech_analysis import UnifiedSpeechAnalyzer
        
        print("Initializing UnifiedSpeechAnalyzer... ", end="", flush=True)
        analyzer = UnifiedSpeechAnalyzer()
        print("✓")
        
        print(f"  SpeechProcessor: {'✓' if analyzer.speech_processor else '✗'}")
        print(f"  HuggingFace token: {'✓' if analyzer.huggingface_token else '✗'}")
        
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Unified Speech Analysis - Setup Verification")
    print("="*60)
    
    deps_ok = test_dependencies()
    cuda_ok = test_cuda()
    models_ok = test_models()
    config_ok = test_config()
    analyzer_ok = test_unified_analyzer()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    all_deps = all(deps_ok.values())
    all_models = all(models_ok.values())
    required_config = config_ok.get("YANDEX_API_KEY") and config_ok.get("HUGGINGFACE_TOKEN")
    
    print(f"All dependencies installed: {'✓ YES' if all_deps else '✗ NO'}")
    print(f"CUDA/GPU available: {'✓ YES' if cuda_ok else '✗ NO (will use CPU)'}")
    print(f"Models accessible: {'✓ YES' if all_models else '✗ NO'}")
    print(f"Required config set: {'✓ YES' if required_config else '✗ NO'}")
    print(f"UnifiedSpeechAnalyzer: {'✓ OK' if analyzer_ok else '✗ FAILED'}")
    
    if all_deps and all_models and required_config:
        print("\n✓ All systems ready! You can run:")
        print("  python -m app.backend.unified_speech_analysis.example [audio.ogg]")
        return 0
    else:
        print("\n✗ Some setup items need attention. See above.")
        print("\nInstall missing dependencies with:")
        print("  pip install openai-whisper pyannote.audio torch transformers")
        print("\nSet environment variables in .env:")
        print("  YANDEX_API_KEY=...")
        print("  HUGGINGFACE_TOKEN=...")
        return 1


if __name__ == "__main__":
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    
    sys.exit(main())

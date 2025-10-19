# verify_cuda.py
# A small script to directly check CUDA support in ctranslate2 and PyTorch.

print("--- Verifying CUDA Environment for AI Libraries ---")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available to PyTorch? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
except ImportError:
    print("❌ PyTorch is not installed. This might be part of the problem.")
except Exception as e:
    print(f"❌ An error occurred while checking PyTorch: {e}")


print("\n" + "-"*20 + "\n")


try:
    import ctranslate2
    print("✅ ctranslate2 library imported successfully.")
    
    # This is the crucial test
    supported_types = ctranslate2.get_supported_compute_types("cuda")
    
    print("\n✅ ctranslate2 successfully detected CUDA.")
    print(f"   Supported compute types on this GPU: {supported_types}")

    if "float16" in supported_types:
        print("\n✅ SUCCESS: 'float16' is officially supported! The main script should work.")
    else:
        print("\n❌ FAILURE: 'float16' is NOT in the list of supported types.")
        print("   This confirms a deep incompatibility in the environment.")
        print("   We will need to use a fallback compute type.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to check ctranslate2 CUDA support.")
    print(f"   This likely means ctranslate2 was not installed with proper CUDA bindings.")
    print(f"   Error details: {e}")
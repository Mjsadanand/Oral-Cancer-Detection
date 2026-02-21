"""
Quick test script to verify installation and setup
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing package imports...\n")
    
    required_packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'PIL': 'Pillow',
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:15} - OK")
        except ImportError:
            print(f"❌ {name:15} - MISSING")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed correctly!")
        return True

def test_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directory structure...\n")
    
    from pathlib import Path
    
    required_dirs = ['static', 'models', 'uploads']
    missing = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ - EXISTS")
        else:
            print(f"⚠️  {dir_name}/ - MISSING (will be created)")
            missing.append(dir_name)
    
    # Create missing directories
    for dir_name in missing:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   Created {dir_name}/")
    
    return True

def test_files():
    """Check if required files exist"""
    print("\n📄 Checking required files...\n")
    
    from pathlib import Path
    
    required_files = {
        'app.py': 'Backend application',
        'train_model.py': 'Training script',
        'requirements.txt': 'Dependencies',
        'static/index.html': 'Frontend UI'
    }
    
    all_exist = True
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path:25} - {description}")
        else:
            print(f"❌ {file_path:25} - MISSING")
            all_exist = False
    
    return all_exist

def test_model():
    """Check if model exists"""
    print("\n🤖 Checking for trained model...\n")
    
    from pathlib import Path
    
    model_path = Path('models/oral_cancer_model.h5')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"⚠️  Model not found: {model_path}")
        print(f"   You need to train the model first:")
        print(f"   1. Prepare your dataset")
        print(f"   2. Run: python train_model.py")
        return False

def test_tensorflow_gpu():
    """Check if TensorFlow can access GPU"""
    print("\n🎮 Checking GPU availability...\n")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No GPU detected - using CPU")
            print("   Training will be slower but will work")
        
        print(f"\n   TensorFlow version: {tf.__version__}")
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🏥 Oral Cancer Detection - System Check")
    print("=" * 60)
    
    tests = [
        ("Package Installation", test_imports),
        ("Directory Structure", test_directories),
        ("Required Files", test_files),
        ("Trained Model", test_model),
        ("GPU Support", test_tensorflow_gpu)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "⚠️  WARNING"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    
    # Next steps
    if not results.get("Trained Model"):
        print("\n📝 Next Steps:")
        print("1. Prepare your dataset in the 'dataset/' folder")
        print("2. Run: python train_model.py")
        print("3. Once trained, run: python app.py")
    else:
        print("\n✅ System ready!")
        print("Run: python app.py")
        print("Then open: http://localhost:8000")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_all_tests()

#!/usr/bin/env python3
"""
Quick Dependency Installer for HyperbolicLearner
Installs only the essential dependencies to get the system running
"""

import subprocess
import sys
import os

def install_package(package, description=""):
    """Install a package with error handling"""
    try:
        print(f"📦 Installing {package}... {description}")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ {package} installed successfully")
            return True
        else:
            print(f"  ⚠️ {package} had issues: {result.stderr.split()[0] if result.stderr else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏳ {package} installation timed out")
        return False
    except Exception as e:
        print(f"  ❌ {package} failed: {e}")
        return False

def main():
    """Install essential dependencies"""
    print("🚀 HyperbolicLearner Quick Dependencies Installer")
    print("=" * 60)
    print("Installing only the most essential packages to get you running...")
    print()
    
    # Essential packages for core functionality
    essential_packages = [
        ("numpy", "Mathematical operations"),
        ("pillow", "Image processing"),
        ("requests", "HTTP requests"),
        ("beautifulsoup4", "Web scraping"),
        ("scikit-learn", "Machine learning"),
        ("pandas", "Data processing"),
        ("fastapi", "API framework"),
        ("uvicorn", "API server"),
        ("selenium", "Web automation"),
        ("pyautogui", "Desktop automation"),
        ("opencv-python", "Computer vision"),
        ("matplotlib", "Plotting and visualization")
    ]
    
    # AI/ML packages (may take longer to install)
    ai_packages = [
        ("transformers", "Hugging Face transformers"),
        ("torch", "PyTorch deep learning"),
        ("sentence-transformers", "Sentence embeddings")
    ]
    
    # Optional packages (install if time permits)
    optional_packages = [
        ("speech-recognition", "Audio processing"),
        ("redis", "Caching database"),
        ("websockets", "Real-time communication")
    ]
    
    successful = []
    failed = []
    
    print("Phase 1: Essential Packages")
    print("-" * 30)
    for package, description in essential_packages:
        if install_package(package, description):
            successful.append(package)
        else:
            failed.append(package)
    
    print("\nPhase 2: AI/ML Packages (may take a few minutes)")
    print("-" * 50)
    for package, description in ai_packages:
        if install_package(package, description):
            successful.append(package)
        else:
            failed.append(package)
    
    print("\nPhase 3: Optional Packages")
    print("-" * 25)
    for package, description in optional_packages:
        if install_package(package, description):
            successful.append(package)
        else:
            failed.append(package)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    success_rate = len(successful) / (len(successful) + len(failed)) * 100 if (successful or failed) else 0
    
    print(f"✅ Successful: {len(successful)} packages")
    print(f"❌ Failed: {len(failed)} packages")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if successful:
        print(f"\n✅ Successfully installed:")
        for pkg in successful:
            print(f"   • {pkg}")
    
    if failed:
        print(f"\n⚠️ Failed to install:")
        for pkg in failed:
            print(f"   • {pkg}")
        print(f"\nYou can manually install these later with:")
        print(f"   pip install {' '.join(failed)}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run: python3 transcendent_launcher.py")
    print("2. Test: python3 src/intelligence/screen_monitor.py")
    print("3. Check system status and start automating!")
    
    if success_rate >= 80:
        print("\n🎉 Your HyperbolicLearner is ready for action!")
    elif success_rate >= 60:
        print("\n⚡ Your HyperbolicLearner has good functionality - some advanced features may be limited")
    else:
        print("\n⚠️ Some core features may be limited - consider installing failed packages manually")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MLX Installation Helper for Apple Silicon

This script helps install the necessary MLX packages for Apple Silicon optimization.
Only supported on macOS with Apple Silicon (M1/M2/M3/M4 chips).
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import argparse

def is_apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.system() == 'Darwin' and platform.machine().startswith('arm')

def get_chip_info():
    """Get Apple chip information."""
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                             capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown Apple chip"

def install_mlx(args):
    """Install MLX and related packages."""
    if not is_apple_silicon():
        print("ERROR: MLX is only supported on macOS with Apple Silicon (M1/M2/M3/M4).")
        print(f"Your system: {platform.system()} {platform.machine()}")
        return False
    
    chip_info = get_chip_info()
    print(f"Detected: {chip_info}")
    print("Proceeding with MLX installation for Apple Silicon...")
    
    # Install basic MLX
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx"], check=True)
        print("✓ Successfully installed MLX!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install MLX: {e}")
        return False
    
    # Install mlx-lm if requested
    if args.with_lm:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "mlx-lm"], check=True)
            print("✓ Successfully installed MLX-LM!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install MLX-LM: {e}")
    
    # Update requirements.txt
    if args.update_requirements:
        try:
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                with open(requirements_file, "r") as f:
                    content = f.read()
                
                # Check if MLX is already in requirements
                if "mlx" not in content:
                    with open(requirements_file, "a") as f:
                        f.write("\nmlx; sys_platform == 'darwin' and platform_machine.startswith('arm')\n")
                    print("✓ Updated requirements.txt with MLX dependency.")
            else:
                print("requirements.txt not found - could not update automatically.")
        except Exception as e:
            print(f"Failed to update requirements.txt: {e}")
    
    print("\nMLX Installation Summary:")
    print("-------------------------")
    print("✓ MLX base package installed")
    if args.with_lm:
        print("✓ MLX-LM package installed")
    if args.update_requirements:
        print("✓ requirements.txt updated")
    
    print("\nTo verify installation, run:")
    print("  python -c 'import mlx; print(mlx.__version__)'")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Install MLX for Apple Silicon optimization")
    parser.add_argument("--with-lm", action="store_true", help="Also install MLX-LM package for language models")
    parser.add_argument("--update-requirements", action="store_true", 
                      help="Update requirements.txt with MLX dependency")
    
    args = parser.parse_args()
    
    if install_mlx(args):
        print("\nMLX installation completed successfully!")
        return 0
    else:
        print("\nMLX installation failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
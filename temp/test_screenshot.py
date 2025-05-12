#!/usr/bin/env python3
"""
Test script to diagnose screenshot capabilities on macOS.
"""

import os
import sys
import platform
import time
from PIL import Image, ImageGrab

# Try different screenshot methods
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")

# Method 1: PIL ImageGrab
print("\n--- Testing PIL ImageGrab ---")
try:
    print(f"PIL version: {Image.__version__}")
    start = time.time()
    img = ImageGrab.grab()
    end = time.time()
    print(f"ImageGrab.grab() success: {img.size}, took {end - start:.3f}s")
    img.save("test_pil.png")
    print("Saved test_pil.png")
except Exception as e:
    print(f"ImageGrab.grab() failed: {e}")

# Method 2: PyAutoGUI
print("\n--- Testing PyAutoGUI ---")
try:
    import pyautogui
    print(f"PyAutoGUI version: {pyautogui.__version__}")
    start = time.time()
    img = pyautogui.screenshot()
    end = time.time()
    print(f"pyautogui.screenshot() success: {img.size}, took {end - start:.3f}s")
    img.save("test_pyautogui.png")
    print("Saved test_pyautogui.png")
except Exception as e:
    print(f"pyautogui.screenshot() failed: {e}")
    print(f"Exception type: {type(e)}")
    import traceback
    traceback.print_exc()

print("\nTest complete. Check the output images.") 
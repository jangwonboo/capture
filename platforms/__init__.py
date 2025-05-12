"""
Platform-specific implementations for window management and screen capture.
"""

import platform
from typing import Dict, Any, Optional

from .macos import MacOSPlatform
from .windows import WindowsPlatform
from .linux import LinuxPlatform

def get_platform_handler():
    """
    Get the appropriate platform handler based on the current operating system.
    
    Returns:
        Platform handler instance
    """
    system = platform.system()
    
    if system == 'Darwin':
        return MacOSPlatform()
    elif system == 'Windows':
        return WindowsPlatform()
    elif system == 'Linux':
        return LinuxPlatform()
    else:
        raise ValueError(f"Unsupported platform: {system}")
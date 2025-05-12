"""
Platform-specific utilities for the e-book page capture tool.
"""

import os
import platform
import logging
import subprocess
from typing import Dict, Any, Tuple, List, Optional
import sys

logger = logging.getLogger('platform')

def get_system_name() -> str:
    """
    Get the current operating system name.
    
    Returns:
        String identifying the operating system ('Windows', 'Darwin', 'Linux')
    """
    return platform.system()

def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon.
    
    Returns:
        Boolean indicating whether the system is using Apple Silicon
    """
    return platform.system() == 'Darwin' and platform.machine().startswith('arm')

def check_command_availability(command: str) -> bool:
    """
    Check if a command is available in the system PATH.
    
    Args:
        command: Command to check
        
    Returns:
        Boolean indicating whether the command is available
    """
    import shutil
    return shutil.which(command) is not None

def run_command(command: List[str], check: bool = False) -> Dict[str, Any]:
    """
    Run a system command and return the result.
    
    Args:
        command: Command to run (as a list of arguments)
        check: Whether to raise an exception on non-zero exit code
        
    Returns:
        Dictionary with command result (returncode, stdout, stderr)
    """
    try:
        logger.debug(f"FLOW: Entering try block in run_command() - running: {' '.join(command)}")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=check
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except Exception as e:
        logger.error(f"Error running command {' '.join(command)}: {e}")
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False,
            'error': str(e)
        }

def open_file(file_path: str) -> bool:
    """
    Open a file with the default application.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Boolean indicating success or failure
    """
    system = get_system_name()
    
    try:
        logger.debug(f"FLOW: Entering try block in open_file() - opening file: {file_path}")
        if system == 'Windows':
            os.startfile(file_path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', file_path], check=False)
        else:  # Linux
            subprocess.run(['xdg-open', file_path], check=False)
        return True
    except Exception as e:
        logger.error(f"Error opening file {file_path}: {e}")
        return False

def get_screen_size() -> Tuple[int, int]:
    """
    Get the primary screen size.
    
    Returns:
        Tuple of (width, height) for the primary screen
    """
    try:
        logger.debug("FLOW: Entering try block in get_screen_size() - determining screen size")
        # First try to use PyAutoGUI if available
        try:
            logger.debug("FLOW: Entering try block to check PyAutoGUI availability")
            import pyautogui
            return pyautogui.size()
        except ImportError:
            logger.debug("PyAutoGUI not available, trying platform-specific methods")
        
        # Platform-specific methods
        system = get_system_name()
        if system == 'Windows':
            try:
                logger.debug("FLOW: Entering try block for Windows screen size detection")
                import ctypes
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return (width, height)
            except Exception as e:
                logger.debug(f"Windows screen size detection failed: {e}")
        elif system == 'Darwin':  # macOS
            try:
                logger.debug("FLOW: Entering try block for macOS Quartz screen size detection")
                import Quartz
                main_display = Quartz.CGMainDisplayID()
                width = Quartz.CGDisplayPixelsWide(main_display)
                height = Quartz.CGDisplayPixelsHigh(main_display)
                return (width, height)
            except ImportError:
                logger.debug("Quartz not available, trying AppleScript")
                # Try using AppleScript
                script = """
                tell application "Finder"
                    set screenResolution to bounds of window of desktop
                    return (item 3 of screenResolution) & "," & (item 4 of screenResolution)
                end tell
                """
                try:
                    logger.debug("FLOW: Entering try block for macOS AppleScript screen size detection")
                    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        dimensions = result.stdout.strip().split(',')
                        if len(dimensions) >= 2:
                            return (int(dimensions[0]), int(dimensions[1]))
                except Exception as e:
                    logger.debug(f"AppleScript screen size detection failed: {e}")
        elif system == 'Linux':
            try:
                logger.debug("FLOW: Entering try block for Linux xrandr screen size detection")
                # Try to use xrandr
                result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    import re
                    pattern = r'current (\d+) x (\d+)'
                    match = re.search(pattern, result.stdout)
                    if match:
                        return (int(match.group(1)), int(match.group(2)))
            except Exception as e:
                logger.debug(f"xrandr screen size detection failed: {e}")
    except Exception as e:
        logger.error(f"Error getting screen size: {e}")
    
    # Fallback to a common resolution
    logger.warning("Could not determine screen size, using default 1920x1080")
    return (1920, 1080)
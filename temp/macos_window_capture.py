#!/usr/bin/env python3
"""
macOS Window Capture Utility

This module provides functions to capture specific application windows on macOS
using the ScreenCaptureKit framework via PyObjC.
"""

import os
import time
import logging
from typing import Tuple, Optional, List, Dict, Any
import json
import subprocess
from PIL import Image
import io
import numpy as np
from PIL import ImageGrab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macos_window_capture")

try:
    # Import PyObjC components
    import Quartz
    from AppKit import NSWorkspace, NSRunningApplication, NSApplicationActivationPolicy
    import Foundation
    import objc
    
    # Import ScreenCaptureKit if available
    try:
        import ScreenCaptureKit
        SCREENCAPTUREKIT_AVAILABLE = True
    except ImportError:
        SCREENCAPTUREKIT_AVAILABLE = False
        logger.warning("ScreenCaptureKit not available. Using fallback methods.")

except ImportError:
    logger.error("PyObjC not properly installed. Please install with: pip install pyobjc")
    SCREENCAPTUREKIT_AVAILABLE = False

def get_running_applications() -> List[Dict[str, Any]]:
    """
    Get a list of all running applications on macOS.
    
    Returns:
        List of dictionaries with application information
    """
    apps = []
    
    try:
        running_apps = NSWorkspace.sharedWorkspace().runningApplications()
        for app in running_apps:
            # Skip background applications without UI
            # Use the numerical value instead of the enum constant
            # NSApplicationActivationPolicyRegular = 0
            if app.activationPolicy() == 0:  # Regular application (with UI)
                app_info = {
                    "name": app.localizedName(),
                    "bundle_id": app.bundleIdentifier(),
                    "pid": app.processIdentifier()
                }
                apps.append(app_info)
    except Exception as e:
        logger.error(f"Error getting running applications: {e}")
    
    return apps

def find_app_by_name(app_name: str) -> Optional[Dict[str, Any]]:
    """
    Find a running application by name (partial match).
    
    Args:
        app_name: Name of the application to find (case-insensitive partial match)
        
    Returns:
        Dictionary with application information or None if not found
    """
    apps = get_running_applications()
    app_name = app_name.lower()
    
    for app in apps:
        if app_name in app["name"].lower():
            return app
    
    return None

def capture_window_with_screencapturekit(app_name: str) -> Optional[Image.Image]:
    """
    Capture a window using ScreenCaptureKit (macOS 12.3+).
    
    Args:
        app_name: Name of the application to capture
        
    Returns:
        PIL Image of the captured window or None if failed
    """
    if not SCREENCAPTUREKIT_AVAILABLE:
        logger.warning("ScreenCaptureKit not available. Using alternative method.")
        return capture_window_with_applescript(app_name)
    
    try:
        # First find the application
        app_info = find_app_by_name(app_name)
        if not app_info:
            logger.error(f"Application '{app_name}' not found")
            return None
            
        logger.info(f"Found application: {app_info['name']} (PID: {app_info['pid']})")
        
        # Use built-in screencapture command with process ID targeting
        # This is a more reliable approach than trying to use the PyObjC ScreenCaptureKit directly
        temp_output_path = f"/tmp/window_capture_{int(time.time())}.png"
        
        # Use the screencapture command with -l flag to capture a specific window by process ID
        cmd = ["screencapture", "-l", str(app_info['pid']), "-o", "-x", temp_output_path]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"screencapture command failed: {result.stderr}")
            return None
            
        # Check if file was created
        if not os.path.exists(temp_output_path):
            logger.error(f"Screenshot file was not created at {temp_output_path}")
            return None
            
        # Load the image file
        try:
            img = Image.open(temp_output_path)
            # Remove the temporary file
            os.remove(temp_output_path)
            return img
        except Exception as e:
            logger.error(f"Error loading captured image: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error capturing window with ScreenCaptureKit: {e}")
        return None

def capture_window_with_applescript(app_name: str) -> Optional[Image.Image]:
    """
    Capture window using AppleScript as a fallback.
    
    Args:
        app_name: Name of the application to capture
        
    Returns:
        PIL Image of the captured window or None if failed
    """
    try:
        temp_output_path = f"/tmp/window_capture_{int(time.time())}.png"
        
        # First, get accurate app name
        get_app_name_script = f'''
        tell application "System Events"
            set appList to (name of every process where name contains "{app_name}")
            if appList is not {{}} then
                return item 1 of appList
            else
                return ""
            end if
        end tell
        '''
        
        app_name_result = subprocess.run(["osascript", "-e", get_app_name_script], 
                                       capture_output=True, text=True, check=False)
        
        exact_app_name = app_name_result.stdout.strip() or app_name
        logger.info(f"Exact app name: {exact_app_name}")
        
        # Get the window bounds
        bounds_script = f'''
        tell application "System Events"
            set theProcess to first process whose name contains "{exact_app_name}"
            set theWindow to first window of theProcess
            set isVisible to visible of theWindow
            set isFrontmost to frontmost of theProcess
            set boundsInfo to position of theWindow & size of theWindow
            return boundsInfo as text
        end tell
        '''
        
        # Try to get window bounds
        bounds_result = subprocess.run(["osascript", "-e", bounds_script], 
                                     capture_output=True, text=True, check=False)
        
        # Check if bounds were retrieved successfully
        if bounds_result.returncode == 0 and bounds_result.stdout.strip():
            try:
                # Parse bounds text into integers
                bounds_text = bounds_result.stdout.strip()
                bounds_text = bounds_text.replace('{', '').replace('}', '')
                bounds = [int(float(x.strip())) for x in bounds_text.split(',')]
                
                if len(bounds) == 4:
                    x, y, x2, y2 = bounds
                    width = x2 - x
                    height = y2 - y
                    logger.info(f"Window bounds: x={x}, y={y}, width={width}, height={height}")
                    
                    # Use screencapture with region
                    capture_cmd = ["screencapture", "-R", f"{x},{y},{width},{height}", "-c", "-o", temp_output_path]
                    subprocess.run(capture_cmd, capture_output=True, check=False)
                    
                    # Check if capture succeeded
                    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                        img = Image.open(temp_output_path)
                        os.remove(temp_output_path)
                        return img
            except Exception as e:
                logger.error(f"Error processing window bounds: {e}")
        
        # If window bounds approach failed, try a direct approach with window ID
        logger.info("Trying direct window ID approach")
        window_id_script = f'''
        tell application "System Events"
            tell application process "{exact_app_name}"
                set frontmost to true
                delay 0.5
            end tell
            
            set frontProcess to first process whose frontmost is true
            if exists window 1 of frontProcess then
                set window_id to id of window 1 of frontProcess
                return window_id
            else
                return ""
            end if
        end tell
        '''
        
        window_id_result = subprocess.run(["osascript", "-e", window_id_script], 
                                       capture_output=True, text=True, check=False)
        
        if window_id_result.returncode == 0 and window_id_result.stdout.strip():
            window_id = window_id_result.stdout.strip()
            logger.info(f"Got window ID: {window_id}")
            
            # Capture using window ID
            subprocess.run(["screencapture", f"-l{window_id}", "-o", temp_output_path], 
                          capture_output=True, check=False)
            
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                img = Image.open(temp_output_path)
                os.remove(temp_output_path)
                return img
        
        # Fall back to full screen capture
        logger.info("Falling back to full screen capture")
        subprocess.run(["screencapture", "-o", temp_output_path], capture_output=True, check=False)
        
        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            img = Image.open(temp_output_path)
            os.remove(temp_output_path)
            return img
            
        logger.error("All capture methods failed")
        return None
    except Exception as e:
        logger.error(f"Error in AppleScript capture: {e}")
        return None

def capture_window_direct(app_name: str) -> Optional[Image.Image]:
    """
    Capture a window using direct terminal commands that work on macOS.
    
    This method uses a two-step process: 
    1. Capture the entire screen
    2. Get the window bounds and crop accordingly
    
    Args:
        app_name: Name of the application to capture
        
    Returns:
        PIL Image of the captured window or None if failed
    """
    try:
        # Find and activate the application
        app_info = find_app_by_name(app_name)
        if not app_info:
            logger.error(f"Application '{app_name}' not found")
            return None
            
        logger.info(f"Found application: {app_info['name']} (PID: {app_info['pid']})")
        
        # First get the exact app name (some apps have different displayed names)
        get_app_name_script = f'''
        tell application "System Events"
            set appList to (name of every process where name contains "{app_name}")
            if appList is not {{}} then
                return item 1 of appList
            else
                return ""
            end if
        end tell
        '''
        
        app_name_result = subprocess.run(["osascript", "-e", get_app_name_script], 
                                       capture_output=True, text=True, check=False)
        
        exact_app_name = app_name_result.stdout.strip() or app_info['name']
        logger.info(f"Exact process name for window capture: {exact_app_name}")
        
        # Step 1: Activate the app and bring it to foreground
        activate_script = f'''
        tell application "{exact_app_name}" to activate
        delay 0.5
        '''
        subprocess.run(["osascript", "-e", activate_script], capture_output=True, check=False)
        
        # Wait for app to become active
        time.sleep(0.5)
        
        # Get more accurate app name that works with Apple Events
        app_process_name = exact_app_name
        
        # Simplified approach to get window info
        window_info_script = f'''
        tell application "System Events"
            tell process "{app_process_name}"
                return properties of window 1
            end tell
        end tell
        '''
        
        # Try to capture using Quartz for pixel-perfect window capture
        try:
            # Capture the entire screen
            full_screen = ImageGrab.grab()
            
            # Get window position using simpler approach
            simple_bounds_script = f'''
            tell application "System Events"
                tell process "{app_process_name}"
                    return position of window 1 & size of window 1
                end tell
            end tell
            '''
            
            bounds_result = subprocess.run(["osascript", "-e", simple_bounds_script], 
                                         capture_output=True, text=True, check=False)
            
            if bounds_result.returncode == 0 and bounds_result.stdout.strip():
                bounds_text = bounds_result.stdout.strip()
                logger.info(f"Window bounds raw data: {bounds_text}")
                
                # The result format is typically "x, y, width, height"
                try:
                    parts = bounds_text.replace('{', '').replace('}', '').split(',')
                    if len(parts) >= 4:
                        x = int(float(parts[0].strip()))
                        y = int(float(parts[1].strip()))
                        width = int(float(parts[2].strip()))
                        height = int(float(parts[3].strip()))
                        
                        logger.info(f"Window position: x={x}, y={y}, width={width}, height={height}")
                        
                        # Crop using the position and size
                        window_image = full_screen.crop((x, y, x + width, y + height))
                        logger.info(f"Successfully cropped window: {window_image.size}")
                        return window_image
                except Exception as e:
                    logger.error(f"Error parsing window position: {e}")
            
            # If bounds parsing failed, try one more approach with direct positions
            try:
                position_script = f'''
                tell application "System Events"
                    tell process "{app_process_name}"
                        set pos to position of window 1
                        set sz to size of window 1
                        return pos & sz
                    end tell
                end tell
                '''
                
                pos_result = subprocess.run(["osascript", "-e", position_script], 
                                          capture_output=True, text=True, check=False)
                
                if pos_result.returncode == 0 and pos_result.stdout.strip():
                    pos_text = pos_result.stdout.strip().replace('{', '').replace('}', '')
                    parts = [float(x.strip()) for x in pos_text.split(',')]
                    
                    if len(parts) >= 4:
                        x = int(parts[0])
                        y = int(parts[1])
                        width = int(parts[2])
                        height = int(parts[3])
                        
                        logger.info(f"Window position (retry): x={x}, y={y}, width={width}, height={height}")
                        
                        # Crop using the position and size
                        window_image = full_screen.crop((x, y, x + width, y + height))
                        logger.info(f"Successfully cropped window (retry): {window_image.size}")
                        return window_image
            except Exception as e:
                logger.error(f"Error in retry crop: {e}")
            
            # If all else fails, return the full screen
            logger.warning("All window cropping methods failed. Returning full screen image.")
            return full_screen
            
        except Exception as e:
            logger.error(f"Error in Quartz capture: {e}")
            return ImageGrab.grab()  # Fallback to full screen
            
    except Exception as e:
        logger.error(f"Error in capture_window_direct: {e}")
        # Attempt to return full screen capture as a last resort
        try:
            return ImageGrab.grab()
        except:
            return None

def capture_app_window(app_name: str) -> Optional[Image.Image]:
    """
    Capture a specific application window using the best available method.
    
    Args:
        app_name: Name of the application to capture
        
    Returns:
        PIL Image of the captured window or None if failed
    """
    # Try direct window capture first - most reliable method
    logger.info(f"Attempting to capture window: {app_name}")
    result = capture_window_direct(app_name)
    if result:
        return result
        
    # Try ScreenCaptureKit method next
    logger.info("Trying ScreenCaptureKit method")
    result = capture_window_with_screencapturekit(app_name)
    if result:
        return result
    
    # Fall back to AppleScript if needed
    logger.info("Falling back to AppleScript method")
    result = capture_window_with_applescript(app_name)
    
    return result

if __name__ == "__main__":
    # Simple test to capture a window
    import sys
    
    if len(sys.argv) > 1:
        app_name = sys.argv[1]
    else:
        print("Available applications:")
        for app in get_running_applications():
            print(f"- {app['name']}")
        app_name = input("Enter application name to capture: ")
    
    img = capture_app_window(app_name)
    if img:
        output_path = f"{app_name.replace(' ', '_')}_capture.png"
        img.save(output_path)
        print(f"Window captured and saved to {output_path}")
    else:
        print("Failed to capture window") 
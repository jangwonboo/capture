"""
macOS-specific implementations for window management and screen capture.
"""

import os
import time
import logging
import subprocess
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image, ImageGrab
import io

logger = logging.getLogger("platform.macos")

class MacOSPlatform:
    """Implementation of platform-specific functionality for macOS."""
    
    def __init__(self):
        """Initialize the macOS platform handler."""
        self.screencapturekit_available = False
        
        # Check for PyObjC availability
        try:
            # Import PyObjC components
            import Quartz
            from AppKit import NSWorkspace, NSRunningApplication
            import Foundation
            
            # Import ScreenCaptureKit if available
            try:
                import ScreenCaptureKit
                self.screencapturekit_available = True
                logger.info("ScreenCaptureKit available for enhanced window capture")
            except ImportError:
                logger.warning("ScreenCaptureKit not available. Using fallback methods.")
                
            self.pyobjc_available = True
            logger.info("PyObjC available for macOS integration")
        except ImportError:
            logger.warning("PyObjC not installed. Some macOS-specific features will be limited.")
            self.pyobjc_available = False
    
    def get_window_by_title(self, window_title: str) -> Optional[Any]:
        """
        Find a window by its title and bring it to the foreground.
        
        Args:
            window_title: Title of the window to find
            
        Returns:
            Boolean indicating success or failure
        """
        if not window_title:
            logger.debug("No window title provided, skipping window focus")
            return False
            
        logger.info(f"Attempting to focus window: '{window_title}'")
        
        try:
            # Use AppleScript to find and focus the application
            script = f'''
            tell application "System Events"
                set isRunning to false
                set appList to name of every process
                repeat with appName in appList
                    if appName contains "{window_title}" then
                        set isRunning to true
                        exit repeat
                    end if
                end repeat
                return isRunning
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.stdout.strip().lower() == "true":
                # If running, bring to front
                logger.debug(f"Found application with title containing '{window_title}'")
                script = f'''
                tell application "System Events"
                    set frontApp to first process whose name contains "{window_title}"
                    set frontAppName to name of frontApp
                    tell application frontAppName
                        activate
                    end tell
                    return frontAppName
                end tell
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                app_name = result.stdout.strip()
                logger.info(f"Successfully activated application: {app_name}")
                return True
            else:
                logger.warning(f"No application found with name containing '{window_title}'")
                return False
        except Exception as e:
            logger.error(f"Error focusing window on macOS: {e}")
            return False
    
    def get_window_rect(self, window_title: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the position and size of a window.
        
        Args:
            window_title: Title of the window
            
        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        try:
            script = f'''
            tell application "System Events"
                tell process whose name contains "{window_title}"
                    set pos to position of window 1
                    set sz to size of window 1
                    return pos & sz
                end tell
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', script], 
                                 capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                bounds_text = result.stdout.strip().replace('{', '').replace('}', '')
                parts = [float(x.strip()) for x in bounds_text.split(',')]
                
                if len(parts) >= 4:
                    x = int(parts[0])
                    y = int(parts[1])
                    width = int(parts[2])
                    height = int(parts[3])
                    return (x, y, width, height)
            return None
            
        except Exception as e:
            logger.error(f"Error getting window rect on macOS: {e}")
            return None
    
    def capture_screen(self, window_title: Optional[str] = None, monitor_index: Optional[int] = None) -> Optional[Image.Image]:
        """
        Capture the screen or a specific window.
        
        Args:
            window_title: Title of the window to capture (if None, captures entire screen or monitor)
            monitor_index: Index of monitor to capture (if None and window_title is None, captures primary monitor)
            
        Returns:
            PIL Image of the captured screen or window
        """
        # If window title is provided, try to capture that specific window
        if window_title:
            # First try specialized capture
            try:
                img = self.capture_app_window(window_title)
                if img:
                    logger.info(f"Successfully captured window with dimensions: {img.size}")
                    return img
            except Exception as e:
                logger.warning(f"Specialized window capture failed: {e}")
                logger.info("Falling back to alternate methods")
            
            # Next try using window position and size
            try:
                window_rect = self.get_window_rect(window_title)
                if window_rect:
                    x, y, width, height = window_rect
                    logger.info(f"Window found at ({x},{y}) with size {width}x{height}")
                    
                    # Try to use screencapture command to capture the window
                    temp_output_path = f"/tmp/window_capture_{int(time.time())}.png"
                    
                    # Capture using region
                    capture_cmd = ["screencapture", "-R", f"{x},{y},{width},{height}", "-o", temp_output_path]
                    subprocess.run(capture_cmd, capture_output=True, check=False)
                    
                    # Check if file was created
                    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                        img = Image.open(temp_output_path)
                        os.remove(temp_output_path)
                        return img
                    
                    # If command failed, try to crop from a full screen capture
                    try:
                        full_screen = ImageGrab.grab()
                        window_img = full_screen.crop((x, y, x + width, y + height))
                        return window_img
                    except Exception as e:
                        logger.error(f"Error capturing screen region: {e}")
                else:
                    logger.warning(f"Could not get position/size for window '{window_title}'")
            except Exception as e:
                logger.error(f"Error with window capture: {e}")
        
        # If we get here, either window capture failed or we're capturing full screen/monitor
        
        # If monitor_index is specified, try to capture that specific monitor
        if monitor_index is not None:
            monitor_info = self.get_monitor_info()
            if 0 <= monitor_index < len(monitor_info['monitors']):
                mon_x, mon_y, mon_width, mon_height = monitor_info['monitors'][monitor_index]
                
                # Convert to absolute coordinates if needed
                if 'virtual_screen' in monitor_info:
                    virt_x, virt_y = monitor_info['virtual_screen'][0], monitor_info['virtual_screen'][1]
                    abs_x = virt_x + mon_x
                    abs_y = virt_y + mon_y
                else:
                    abs_x, abs_y = mon_x, mon_y
                    
                logger.info(f"Capturing monitor {monitor_index} at ({abs_x},{abs_y}) with size {mon_width}x{mon_height}")
                
                # Use screencapture command
                temp_output_path = f"/tmp/monitor_capture_{int(time.time())}.png"
                capture_cmd = ["screencapture", "-R", f"{abs_x},{abs_y},{mon_width},{mon_height}", "-o", temp_output_path]
                result = subprocess.run(capture_cmd, capture_output=True, check=False)
                
                if result.returncode == 0 and os.path.exists(temp_output_path):
                    img = Image.open(temp_output_path)
                    os.remove(temp_output_path)
                    return img
                
                # If that fails, try to use ImageGrab and crop
                try:
                    full_screen = ImageGrab.grab()
                    monitor_img = full_screen.crop((abs_x, abs_y, abs_x + mon_width, abs_y + mon_height))
                    return monitor_img
                except Exception as e:
                    logger.error(f"Error capturing monitor with ImageGrab: {e}")
        
        # Default: capture entire screen
        try:
            return ImageGrab.grab()
        except Exception as e:
            logger.error(f"Error capturing screen with ImageGrab: {e}")
            
            # Last attempt using screencapture command
            try:
                temp_output_path = f"/tmp/screen_capture_{int(time.time())}.png"
                subprocess.run(["screencapture", "-x", temp_output_path], check=False)
                if os.path.exists(temp_output_path):
                    img = Image.open(temp_output_path)
                    os.remove(temp_output_path)
                    return img
            except Exception as e2:
                logger.error(f"All screen capture methods failed: {e2}")
            
            # Return a blank image as last resort
            logger.error("Creating blank image as fallback")
            return Image.new('RGB', (1920, 1080), color='white')

    def resize_window(self, window_title: str, width: int, height: int, x: int, y: int) -> bool:
        """
        Resize and position a window.
        
        Args:
            window_title: Title of the window to resize
            width: Target width in pixels
            height: Target height in pixels
            x: Target x position
            y: Target y position
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Use AppleScript to resize window
            script = f'''
            tell application "System Events"
                set appProcess to first process whose name contains "{window_title}"
                set frontmost of appProcess to true
                delay 0.5
                
                tell window 1 of appProcess
                    set position to {{{x}, {y}}}
                    set size to {{{width}, {height}}}
                end tell
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully resized window to {width}x{height} at ({x},{y})")
                return True
            else:
                logger.error(f"Failed to resize window: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error resizing window: {e}")
            return False
    
    def get_monitor_info(self) -> Dict[str, Any]:
        """
        Get information about all monitors in a multi-monitor setup.
        
        Returns:
            Dictionary containing monitor information
        """
        try:
            # Use PyObjC to get display information if available
            if self.pyobjc_available:
                try:
                    import Quartz
                    
                    monitors = []
                    primary_width, primary_height = 0, 0
                    
                    # Get all displays
                    main_display = Quartz.CGMainDisplayID()
                    all_displays = Quartz.CGGetActiveDisplayList(10, None, None)[1]
                    
                    for i, display in enumerate(all_displays):
                        # Get display bounds
                        bounds = Quartz.CGDisplayBounds(display)
                        x, y = bounds.origin.x, bounds.origin.y
                        width, height = bounds.size.width, bounds.size.height
                        
                        # Check if this is the primary display
                        is_primary = (display == main_display)
                        
                        monitors.append({
                            'x': int(x),
                            'y': int(y),
                            'width': int(width),
                            'height': int(height),
                            'is_primary': is_primary
                        })
                        
                        # Store primary monitor dimensions
                        if is_primary:
                            primary_width = int(width)
                            primary_height = int(height)
                    
                    # Calculate the total dimensions
                    min_x = min(m['x'] for m in monitors)
                    min_y = min(m['y'] for m in monitors)
                    max_x = max(m['x'] + m['width'] for m in monitors)
                    max_y = max(m['y'] + m['height'] for m in monitors)
                    
                    # Create result
                    total_width = max_x - min_x
                    max_height = max_y - min_y
                    
                    # Normalize coordinates
                    for m in monitors:
                        m['x'] -= min_x
                        m['y'] -= min_y
                    
                    return {
                        'primary': (primary_width, primary_height),
                        'all': (total_width, max_height),
                        'monitors': [(m['x'], m['y'], m['width'], m['height']) for m in monitors],
                        'primary_index': next((i for i, m in enumerate(monitors) if m['is_primary']), 0),
                        'virtual_screen': (min_x, min_y, max_x - min_x, max_y - min_y)
                    }
                except Exception as e:
                    logger.error(f"Error getting PyObjC display info: {e}")
            
            # Use AppleScript as fallback
            try:
                script = """
                tell application "System Events"
                    set screenWidth to do shell script "system_profiler SPDisplaysDataType | grep Resolution | awk '{print $2}' | head -1"
                    set screenHeight to do shell script "system_profiler SPDisplaysDataType | grep Resolution | awk '{print $4}' | head -1"
                    return screenWidth & "," & screenHeight
                end tell
                """
                
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 2:
                        width, height = int(parts[0]), int(parts[1])
                        return {
                            'primary': (width, height),
                            'all': (width, height),
                            'monitors': [(0, 0, width, height)],
                            'primary_index': 0,
                            'virtual_screen': (0, 0, width, height)
                        }
            except Exception as e:
                logger.warning(f"Failed to get screen info with AppleScript: {e}")
        except Exception as e:
            logger.error(f"Error getting display info on macOS: {e}")
        
        # Fall back to PyAutoGUI
        try:
            import pyautogui
            width, height = pyautogui.size()
            return {
                'primary': (width, height),
                'all': (width, height),
                'monitors': [(0, 0, width, height)],
                'primary_index': 0,
                'virtual_screen': (0, 0, width, height)
            }
        except ImportError:
            # Return a sensible default as last resort
            logger.error("Could not determine screen size, using default values")
            return {
                'primary': (1920, 1080),
                'all': (1920, 1080),
                'monitors': [(0, 0, 1920, 1080)],
                'primary_index': 0,
                'virtual_screen': (0, 0, 1920, 1080)
            }
    
    def send_keystroke(self, key_name: str) -> bool:
        """
        Send a keystroke to the active application.
        
        Args:
            key_name: Name of the key to press
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # First try PyAutoGUI
            try:
                import pyautogui
                # Handle common navigation keys
                if key_name in ('right', 'left', 'space', 'enter', 'pagedown', 'pageup'):
                    pyautogui.press(key_name)
                # Handle arrow keys with different notation
                elif key_name == 'right_arrow':
                    pyautogui.press('right')
                elif key_name == 'left_arrow':
                    pyautogui.press('left')
                elif key_name == 'down_arrow':
                    pyautogui.press('down')
                elif key_name == 'up_arrow':
                    pyautogui.press('up')
                # Handle any other key
                else:
                    pyautogui.press(key_name)
                
                logger.debug(f"Keystroke '{key_name}' sent successfully using PyAutoGUI")
                return True
            except ImportError:
                logger.debug("PyAutoGUI not available, using AppleScript for keystroke")
            
            # Fall back to AppleScript
            key_map = {
                'right': 'right arrow',
                'left': 'left arrow',
                'up': 'up arrow',
                'down': 'down arrow',
                'right_arrow': 'right arrow',
                'left_arrow': 'left arrow',
                'down_arrow': 'down arrow',
                'up_arrow': 'up arrow',
                'pagedown': 'page down',
                'pageup': 'page up',
                'enter': 'return',
                'space': 'space'
            }
            
            applescript_key = key_map.get(key_name, key_name)
            
            script = f'''
            tell application "System Events"
                key code (key code of "{applescript_key}")
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug(f"Keystroke '{key_name}' sent successfully using AppleScript")
                return True
            else:
                logger.error(f"AppleScript keystroke failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error sending keystroke '{key_name}': {e}")
            return False
    
    def capture_app_window(self, app_name: str) -> Optional[Image.Image]:
        """
        Capture a specific application window using the best available method.
        
        Args:
            app_name: Name of the application to capture
            
        Returns:
            PIL Image of the captured window or None if failed
        """
        # Try to use specialized capture module if available
        if self.screencapturekit_available:
            return self._capture_window_with_screencapturekit(app_name)
        else:
            return self._capture_window_with_applescript(app_name)
    
    def _capture_window_with_screencapturekit(self, app_name: str) -> Optional[Image.Image]:
        """
        Capture a window using ScreenCaptureKit (macOS 12.3+).
        
        Args:
            app_name: Name of the application to capture
            
        Returns:
            PIL Image of the captured window or None if failed
        """
        try:
            # Find the application
            app_info = self._find_app_by_name(app_name)
            if not app_info:
                logger.error(f"Application '{app_name}' not found")
                return None
                
            logger.info(f"Found application: {app_info['name']} (PID: {app_info['pid']})")
            
            # Use built-in screencapture command with process ID targeting
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
    
    def _capture_window_with_applescript(self, app_name: str) -> Optional[Image.Image]:
        """
        Capture window using AppleScript as a fallback.
        
        Args:
            app_name: Name of the application to capture
            
        Returns:
            PIL Image of the captured window or None if failed
        """
        try:
            temp_output_path = f"/tmp/window_capture_{int(time.time())}.png"
            
            # Get accurate app name
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
                        x, y, width, height = bounds
                        logger.info(f"Window bounds: x={x}, y={y}, width={width}, height={height}")
                        
                        # Use screencapture with region
                        capture_cmd = ["screencapture", "-R", f"{x},{y},{width},{height}", "-o", temp_output_path]
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
                    
            # Final fallback - capture full screen and hope the window is focused
            logger.warning("Window-specific capture failed, capturing full screen")
            subprocess.run(["screencapture", "-o", temp_output_path], 
                          capture_output=True, check=False)
                          
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                img = Image.open(temp_output_path)
                os.remove(temp_output_path)
                return img
                
            return None
            
        except Exception as e:
            logger.error(f"Error capturing window with AppleScript: {e}")
            return None
    
    def _get_running_applications(self) -> List[Dict[str, Any]]:
        """
        Get a list of all running applications.
        
        Returns:
            List of dictionaries with application information
        """
        apps = []
        
        if not self.pyobjc_available:
            # Fallback to AppleScript if PyObjC is not available
            try:
                script = '''
                tell application "System Events"
                    set appList to name of every process
                    return appList
                end tell
                '''
                
                result = subprocess.run(['osascript', '-e', script], 
                                     capture_output=True, text=True, check=False)
                
                if result.returncode == 0 and result.stdout.strip():
                    app_names = result.stdout.strip().split(", ")
                    for name in app_names:
                        name = name.strip()
                        if name:
                            apps.append({
                                'name': name,
                                'bundle_id': '',
                                'pid': 0
                            })
            except Exception as e:
                logger.error(f"Error getting running applications with AppleScript: {e}")
            
            return apps
        
        try:
            from AppKit import NSWorkspace
            
            running_apps = NSWorkspace.sharedWorkspace().runningApplications()
            for app in running_apps:
                # Skip background applications without UI
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
    
    def _find_app_by_name(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a running application by name (partial match).
        
        Args:
            app_name: Name of the application to find (case-insensitive partial match)
            
        Returns:
            Dictionary with application information or None if not found
        """
        apps = self._get_running_applications()
        app_name = app_name.lower()
        
        for app in apps:
            if app_name in app["name"].lower():
                return app
        
        return None
"""
Linux-specific implementations for window management and screen capture.
"""

import os
import logging
import subprocess
import re
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image

logger = logging.getLogger("platform.linux")

class LinuxPlatform:
    """Implementation of platform-specific functionality for Linux."""
    
    def __init__(self):
        """Initialize the Linux platform handler."""
        # Check for required tools
        self.xdotool_available = self._check_command('xdotool')
        if not self.xdotool_available:
            logger.warning("xdotool not installed. Window management features will be limited.")
            logger.info("Install xdotool with: sudo apt-get install xdotool")
            
        self.xrandr_available = self._check_command('xrandr')
        if not self.xrandr_available:
            logger.warning("xrandr not installed. Monitor detection will be limited.")
            
        self.scrot_available = self._check_command('scrot')
        if not self.scrot_available:
            logger.warning("scrot not installed. Screen capture may be limited.")
            logger.info("Install scrot with: sudo apt-get install scrot")
    
    def _check_command(self, command: str) -> bool:
        """
        Check if a command is available in the system.
        
        Args:
            command: Command to check
            
        Returns:
            Boolean indicating if the command is available
        """
        try:
            result = subprocess.run(['which', command], capture_output=True, text=True)
            return result.returncode == 0 and result.stdout.strip() != ''
        except Exception:
            return False
    
    def get_window_by_title(self, window_title: str) -> Optional[str]:
        """
        Find a window by its title and bring it to the foreground.
        
        Args:
            window_title: Title of the window to find
            
        Returns:
            Window ID as string or None if not found
        """
        if not window_title:
            logger.debug("No window title provided, skipping window focus")
            return None
            
        logger.info(f"Attempting to focus window: '{window_title}'")
        
        if not self.xdotool_available:
            logger.error("xdotool not available. Cannot focus window.")
            return None
        
        try:
            # Find windows matching the title
            result = subprocess.run(
                ['xdotool', 'search', '--name', window_title], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Get the first window ID
                window_id = result.stdout.strip().split('\n')[0]
                
                # Activate the window
                subprocess.run(['xdotool', 'windowactivate', window_id], check=False)
                
                logger.info(f"Successfully activated window with ID {window_id}")
                return window_id
            else:
                logger.warning(f"No window found with title containing '{window_title}'")
                return None
        except Exception as e:
            logger.error(f"Error finding window on Linux: {e}")
            return None
    
    def get_window_rect(self, window_title: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the position and size of a window.
        
        Args:
            window_title: Title of the window
            
        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        if not self.xdotool_available:
            logger.error("xdotool not available. Cannot get window position.")
            return None
            
        try:
            # First get the window ID
            result = subprocess.run(['xdotool', 'search', '--name', window_title], 
                                 capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]
                
                # Get window geometry
                result = subprocess.run(['xdotool', 'getwindowgeometry', window_id], 
                                     capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    # Parse position and size
                    pos_match = re.search(r'Position: (\d+),(\d+)', result.stdout)
                    size_match = re.search(r'Geometry: (\d+)x(\d+)', result.stdout)
                    
                    if pos_match and size_match:
                        x, y = map(int, pos_match.groups())
                        width, height = map(int, size_match.groups())
                        return (x, y, width, height)
            return None
            
        except Exception as e:
            logger.error(f"Error getting window rect on Linux: {e}")
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
        if window_title and self.xdotool_available:
            try:
                # First, get window position and size
                window_rect = self.get_window_rect(window_title)
                if window_rect:
                    x, y, width, height = window_rect
                    logger.info(f"Window found at ({x},{y}) with size {width}x{height}")
                    
                    # Get window ID
                    window_id_result = subprocess.run(['xdotool', 'search', '--name', window_title], 
                                                  capture_output=True, text=True, check=False)
                    
                    if window_id_result.returncode == 0 and window_id_result.stdout.strip():
                        window_id = window_id_result.stdout.strip().split('\n')[0]
                        
                        # Activate window (bring to front)
                        subprocess.run(['xdotool', 'windowactivate', window_id], check=False)
                        
                        # If scrot is available, use it for window capture
                        if self.scrot_available:
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                                temp_path = temp_file.name
                            
                            # Capture the window with scrot
                            scrot_result = subprocess.run(
                                ['scrot', '-u', temp_path], 
                                capture_output=True, 
                                check=False
                            )
                            
                            if scrot_result.returncode == 0 and os.path.exists(temp_path):
                                img = Image.open(temp_path)
                                os.unlink(temp_path)
                                return img
                        
                        # Fall back to taking screenshot with PyAutoGUI and cropping
                        try:
                            import pyautogui
                            screenshot = pyautogui.screenshot()
                            window_img = screenshot.crop((x, y, x + width, y + height))
                            return window_img
                        except ImportError:
                            logger.warning("PyAutoGUI not available for screen capture")
                
                # If we reached here, window capture failed
                logger.warning("Window capture failed, falling back to full screen capture")
            except Exception as e:
                logger.error(f"Error capturing window: {e}")
        
        # If we get here, either window capture failed or we're capturing full screen/monitor
        
        # If monitor_index is specified, try to capture that specific monitor
        if monitor_index is not None and self.xrandr_available:
            try:
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
                    
                    # Use scrot for screen capture if available
                    if self.scrot_available:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Capture the entire screen with scrot
                        scrot_result = subprocess.run(
                            ['scrot', temp_path], 
                            capture_output=True, 
                            check=False
                        )
                        
                        if scrot_result.returncode == 0 and os.path.exists(temp_path):
                            img = Image.open(temp_path)
                            monitor_img = img.crop((abs_x, abs_y, abs_x + mon_width, abs_y + mon_height))
                            os.unlink(temp_path)
                            return monitor_img
                    
                    # Fall back to PyAutoGUI
                    try:
                        import pyautogui
                        screenshot = pyautogui.screenshot()
                        monitor_img = screenshot.crop((abs_x, abs_y, abs_x + mon_width, abs_y + mon_height))
                        return monitor_img
                    except ImportError:
                        logger.warning("PyAutoGUI not available for screen capture")
            except Exception as e:
                logger.error(f"Error capturing monitor: {e}")
        
        # Default: capture entire screen
        try:
            # Use scrot if available
            if self.scrot_available:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                subprocess.run(['scrot', temp_path], check=False)
                
                if os.path.exists(temp_path):
                    img = Image.open(temp_path)
                    os.unlink(temp_path)
                    return img
            
            # Try PyAutoGUI
            import pyautogui
            return pyautogui.screenshot()
        except ImportError:
            logger.error("PyAutoGUI not available for screenshot")
            
            # Try PIL's ImageGrab as a fallback
            try:
                from PIL import ImageGrab
                return ImageGrab.grab()
            except Exception as e:
                logger.error(f"All screen capture methods failed: {e}")
                
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
        if not self.xdotool_available:
            logger.error("xdotool not available. Cannot resize window.")
            return False
        
        try:
            # First get the window ID
            result = subprocess.run(['xdotool', 'search', '--name', window_title], 
                                 capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]
                
                # Move the window
                subprocess.run(['xdotool', 'windowmove', window_id, str(x), str(y)], check=False)
                
                # Resize the window
                subprocess.run(['xdotool', 'windowsize', window_id, str(width), str(height)], check=False)
                
                logger.info(f"Successfully resized window to {width}x{height} at ({x},{y})")
                return True
            else:
                logger.error(f"Window '{window_title}' not found for resize operation")
                return False
        except Exception as e:
            logger.error(f"Error resizing window on Linux: {e}")
            return False
    
    def get_monitor_info(self) -> Dict[str, Any]:
        """
        Get information about all monitors in a multi-monitor setup.
        
        Returns:
            Dictionary containing monitor information
        """
        try:
            if self.xrandr_available:
                # Use xrandr command to get display information
                result = subprocess.run(['xrandr', '--listmonitors'], capture_output=True, text=True, check=False)
                
                if result.returncode == 0 and result.stdout.strip():
                    monitors = []
                    lines = result.stdout.strip().split('\n')[1:]  # Skip the first line
                    
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            # Extract dimensions and position
                            # Format is typically: "0: +*HDMI-1 1920/509x1080/286+0+0"
                            # Where +* indicates primary
                            is_primary = '*' in parts[1]
                            geo_parts = parts[2].split('+')
                            if len(geo_parts) >= 3:
                                resolution = geo_parts[0].split('x')
                                if '/' in resolution[0]:
                                    resolution[0] = resolution[0].split('/')[0]
                                if '/' in resolution[1]:
                                    resolution[1] = resolution[1].split('/')[0]
                                
                                width, height = int(resolution[0]), int(resolution[1])
                                x, y = int(geo_parts[1]), int(geo_parts[2])
                                
                                monitors.append({
                                    'x': x,
                                    'y': y,
                                    'width': width,
                                    'height': height,
                                    'is_primary': is_primary
                                })
                    
                    if monitors:
                        # Find primary monitor dimensions
                        primary = next((m for m in monitors if m['is_primary']), monitors[0])
                        primary_width, primary_height = primary['width'], primary['height']
                        
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
                        
                # Alternative approach using xrandr --current
                result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    monitors = []
                    primary_info = None
                    
                    # Parse xrandr output
                    # Example line: "HDMI-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis) 598mm x 336mm"
                    connector_pattern = r'(\S+) connected(?: primary)? (\d+)x(\d+)\+(\d+)\+(\d+)'
                    primary_pattern = r'(\S+) connected primary'
                    
                    for line in result.stdout.strip().split('\n'):
                        connector_match = re.search(connector_pattern, line)
                        if connector_match:
                            name = connector_match.group(1)
                            width = int(connector_match.group(2))
                            height = int(connector_match.group(3))
                            x = int(connector_match.group(4))
                            y = int(connector_match.group(5))
                            
                            is_primary = re.search(primary_pattern, line) is not None
                            
                            monitor_info = {
                                'name': name,
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'is_primary': is_primary
                            }
                            
                            monitors.append(monitor_info)
                            
                            if is_primary:
                                primary_info = monitor_info
                    
                    if monitors:
                        # Use the first monitor as primary if none marked as primary
                        if primary_info is None:
                            primary_info = monitors[0]
                            primary_info['is_primary'] = True
                        
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
                            'primary': (primary_info['width'], primary_info['height']),
                            'all': (total_width, max_height),
                            'monitors': [(m['x'], m['y'], m['width'], m['height']) for m in monitors],
                            'primary_index': next((i for i, m in enumerate(monitors) if m['is_primary']), 0),
                            'virtual_screen': (min_x, min_y, max_x - min_x, max_y - min_y)
                        }
            
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
                logger.warning("PyAutoGUI not available for screen size detection")
        except Exception as e:
            logger.error(f"Error in get_monitor_info: {e}")
        
        # Return a sensible default as last resort
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
        # Try to use xdotool if available
        if self.xdotool_available:
            try:
                # Map key names to xdotool key names
                key_map = {
                    'right': 'Right',
                    'left': 'Left',
                    'up': 'Up',
                    'down': 'Down',
                    'right_arrow': 'Right',
                    'left_arrow': 'Left',
                    'down_arrow': 'Down',
                    'up_arrow': 'Up',
                    'pagedown': 'Page_Down',
                    'pageup': 'Page_Up',
                    'enter': 'Return',
                    'space': 'space'
                }
                
                xdotool_key = key_map.get(key_name, key_name)
                
                # Send the key press
                result = subprocess.run(['xdotool', 'key', xdotool_key], 
                                     capture_output=True, check=False)
                
                if result.returncode == 0:
                    logger.debug(f"Keystroke '{key_name}' sent successfully using xdotool")
                    return True
                else:
                    logger.error(f"xdotool keystroke failed: {result.stderr}")
                    
                    # Try fallback key
                    if key_name != 'Right':
                        logger.warning("Trying fallback to Right key")
                        result = subprocess.run(['xdotool', 'key', 'Right'], 
                                             capture_output=True, check=False)
                        if result.returncode == 0:
                            logger.info("Fallback keystroke successful")
                            return True
                    
                    return False
            except Exception as e:
                logger.error(f"Error sending keystroke with xdotool: {e}")
        
        # Try to use PyAutoGUI as a fallback
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
            logger.error("Neither xdotool nor PyAutoGUI is available for sending keystrokes")
            return False
        except Exception as e:
            logger.error(f"Error sending keystroke with PyAutoGUI: {e}")
            return False
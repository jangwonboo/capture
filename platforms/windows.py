"""
Windows-specific implementations for window management and screen capture.
"""

import os
import logging
import subprocess
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
import time

logger = logging.getLogger("platform.windows")

class WindowsPlatform:
    """Implementation of platform-specific functionality for Windows."""
    
    def __init__(self):
        """Initialize the Windows platform handler."""
        self.win32_available = False
        
        # Check for win32 modules availability
        try:
            import win32gui
            import win32con
            import win32ui
            import ctypes
            from ctypes import windll
            
            self.win32_available = True
            logger.info("Win32 modules available for Windows integration")
        except ImportError:
            logger.warning("Win32 modules not installed. Some Windows-specific features will be limited.")
            logger.info("Install pywin32 package for full Windows integration: pip install pywin32")
    
    def get_window_by_title(self, window_title: str) -> Optional[Any]:
        """
        Find a window by its title and bring it to the foreground.
        
        Args:
            window_title: Title of the window to find
            
        Returns:
            Window handle or None if not found
        """
        if not window_title:
            logger.debug("No window title provided, skipping window focus")
            return None
            
        logger.info(f"Attempting to focus window: '{window_title}'")
        
        if self.win32_available:
            try:
                import win32gui
                
                # Find windows containing the title string
                def window_enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd) and window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                        results.append(hwnd)
                    return True
                    
                window_handles = []
                win32gui.EnumWindows(window_enum_callback, window_handles)
                
                if window_handles:
                    # Get the first window that matches
                    hwnd = window_handles[0]
                    window_text = win32gui.GetWindowText(hwnd)
                    logger.info(f"Found window '{window_text}' with handle {hwnd}")
                    # Bring window to front
                    win32gui.SetForegroundWindow(hwnd)
                    return hwnd
                
                logger.warning(f"No window found with title containing '{window_title}'")
                return None
            except Exception as e:
                logger.error(f"Error finding window on Windows: {e}")
                return None
        else:
            logger.warning("Win32 modules not available. Cannot focus window.")
            return None
    
    def get_window_rect(self, window_title: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the position and size of a window.
        
        Args:
            window_title: Title of the window
            
        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        if self.win32_available:
            try:
                import win32gui
                
                def callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd) and window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                        rect = win32gui.GetWindowRect(hwnd)
                        x, y, x_end, y_end = rect
                        results.append((x, y, x_end - x, y_end - y))
                    return True
                
                results = []
                win32gui.EnumWindows(callback, results)
                
                if results:
                    return results[0]
                return None
            except Exception as e:
                logger.error(f"Error getting window rect on Windows: {e}")
                return None
        else:
            logger.warning("Win32 modules not available. Cannot get window position.")
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
        if window_title and self.win32_available:
            try:
                import win32gui
                import win32ui
                import win32con
                from ctypes import windll
                
                # Find window by title
                hwnd = self.get_window_by_title(window_title)
                if hwnd:
                    # Get window dimensions
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    width, height = right - left, bottom - top
                    
                    # Get window client area for more accurate capture
                    border_width = win32gui.GetSystemMetrics(32)
                    title_height = win32gui.GetSystemMetrics(33)
                    client_rect = win32gui.GetClientRect(hwnd)
                    client_width, client_height = client_rect[2], client_rect[3]
                    
                    # Adjust client area to window coordinates
                    x, y = win32gui.ClientToScreen(hwnd, (0, 0))
                    x1, y1 = win32gui.ClientToScreen(hwnd, (client_width, client_height))
                    
                    # Create a device context and a bitmap
                    try:
                        hdc = win32gui.GetWindowDC(hwnd)
                        src_dc = win32ui.CreateDCFromHandle(hdc)
                        mem_dc = src_dc.CreateCompatibleDC()
                        
                        bitmap = win32ui.CreateBitmap()
                        bitmap.CreateCompatibleBitmap(src_dc, x1 - x, y1 - y)
                        mem_dc.SelectObject(bitmap)
                        mem_dc.BitBlt((0, 0), (x1 - x, y1 - y), src_dc, (0, 0), win32con.SRCCOPY)
                        
                        # Convert bitmap to PIL Image
                        bmp_info = bitmap.GetInfo()
                        bmp_bits = bitmap.GetBitmapBits(True)
                        img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_bits, 'raw', 'BGRX', 0, 1)
                        
                        # Clean up resources
                        mem_dc.DeleteDC()
                        src_dc.DeleteDC()
                        win32gui.ReleaseDC(hwnd, hdc)
                        win32gui.DeleteObject(bitmap.GetHandle())
                        
                        return img
                    except Exception as e:
                        logger.error(f"Error capturing window using Windows API: {e}")
                        # Fall back to region-based capture using PyAutoGUI
                        try:
                            import pyautogui
                            logger.info("Falling back to PyAutoGUI region capture")
                            img = pyautogui.screenshot(region=(x, y, x1 - x, y1 - y))
                            return img
                        except ImportError:
                            logger.error("PyAutoGUI not available for fallback capture")
                else:
                    logger.warning(f"Window '{window_title}' not found, falling back to full screen capture")
            except Exception as e:
                logger.error(f"Error during window capture: {e}")
        
        # If we get here, either window capture failed or we're capturing full screen/monitor
        
        # If monitor_index is specified, try to capture that specific monitor
        if monitor_index is not None and self.win32_available:
            try:
                import win32gui
                import win32ui
                import win32con
                from ctypes import windll
                
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
                    
                    # Create device context for the entire screen
                    hdc = win32gui.GetDC(0)
                    src_dc = win32ui.CreateDCFromHandle(hdc)
                    mem_dc = win32ui.CreateDCFromHandle(hdc)
                    mem_dc.CreateCompatibleDC(src_dc)
                    
                    # Create bitmap to hold the capture
                    bitmap = win32ui.CreateBitmap()
                    bitmap.CreateCompatibleBitmap(src_dc, mon_width, mon_height)
                    mem_dc.SelectObject(bitmap)
                    
                    # Copy screen to our bitmap
                    mem_dc.BitBlt((0, 0), (mon_width, mon_height), src_dc, (abs_x, abs_y), win32con.SRCCOPY)
                    
                    # Convert bitmap to PIL Image
                    bmp_info = bitmap.GetInfo()
                    bmp_bits = bitmap.GetBitmapBits(True)
                    img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_bits, 'raw', 'BGRX', 0, 1)
                    
                    # Clean up resources
                    win32gui.DeleteObject(bitmap.GetHandle())
                    mem_dc.DeleteDC()
                    src_dc.DeleteDC()
                    win32gui.ReleaseDC(0, hdc)
                    
                    return img
            except Exception as e:
                logger.error(f"Error capturing specific monitor: {e}")
                # Fall back to PyAutoGUI region capture
                try:
                    import pyautogui
                    return pyautogui.screenshot(region=(abs_x, abs_y, mon_width, mon_height))
                except ImportError:
                    logger.error("PyAutoGUI not available for fallback capture")
        
        # Default: capture entire screen
        try:
            import pyautogui
            return pyautogui.screenshot()
        except ImportError:
            logger.error("PyAutoGUI not available for screenshot")
            
            # Try ImageGrab as a fallback
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
        if self.win32_available:
            try:
                import win32gui
                import win32con
                
                # Find the window
                hwnd = self.get_window_by_title(window_title)
                if not hwnd:
                    logger.error(f"Window '{window_title}' not found")
                    return False
                
                # Resize and position window
                win32gui.MoveWindow(hwnd, x, y, width, height, True)
                logger.info(f"Successfully resized window to {width}x{height} at ({x},{y})")
                return True
            except Exception as e:
                logger.error(f"Error resizing window on Windows: {e}")
                return False
        else:
            logger.error("Win32 modules not available for window resizing")
            return False
    
    def get_monitor_info(self) -> Dict[str, Any]:
        """
        Get information about all monitors in a multi-monitor setup.
        
        Returns:
            Dictionary containing monitor information
        """
        try:
            if self.win32_available:
                try:
                    import ctypes
                    from ctypes import windll, wintypes
                    
                    # Define necessary structures and functions
                    class RECT(ctypes.Structure):
                        _fields_ = [('left', ctypes.c_long),
                                   ('top', ctypes.c_long),
                                   ('right', ctypes.c_long),
                                   ('bottom', ctypes.c_long)]
                    
                    monitors = []
                    primary_width, primary_height = 0, 0
                    
                    # Callback function for EnumDisplayMonitors
                    def callback(monitor, dc, rect, data):
                        # Get monitor info
                        monitor_info = wintypes.MONITORINFOEX()
                        monitor_info.cbSize = ctypes.sizeof(monitor_info)
                        windll.user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info))
                        
                        # Monitor coordinates
                        x = monitor_info.rcMonitor.left
                        y = monitor_info.rcMonitor.top
                        width = monitor_info.rcMonitor.right - monitor_info.rcMonitor.left
                        height = monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top
                        
                        # Check if this is the primary monitor
                        is_primary = (monitor_info.dwFlags & 1) != 0
                        
                        monitors.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'is_primary': is_primary
                        })
                        
                        # Store primary monitor dimensions
                        nonlocal primary_width, primary_height
                        if is_primary:
                            primary_width = width
                            primary_height = height
                        
                        return True
                    
                    # Enumerate all display monitors
                    windll.user32.EnumDisplayMonitors(None, None, ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_void_p)(callback), None)
                    
                    # Calculate the total dimensions
                    min_x = min(m['x'] for m in monitors)
                    min_y = min(m['y'] for m in monitors)
                    max_x = max(m['x'] + m['width'] for m in monitors)
                    max_y = max(m['y'] + m['height'] for m in monitors)
                    
                    # Create result
                    total_width = max_x - min_x
                    max_height = max_y - min_y
                    
                    # Normalize coordinates (make them relative to top-left corner of the virtual screen)
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
                    logger.error(f"Error getting monitor info using Win32: {e}")
            
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
                logger.error("PyAutoGUI not available for screen size detection")
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
        try:
            # Try to use PyAutoGUI
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
            
            logger.debug(f"Keystroke '{key_name}' sent successfully")
            return True
        except ImportError:
            logger.error("PyAutoGUI not available for sending keystrokes")
            return False
        except Exception as e:
            logger.error(f"Error sending keystroke '{key_name}': {e}")
            
            # Try fallback if original keystroke fails
            if key_name != 'right':
                logger.warning(f"Trying fallback to 'right' arrow key")
                try:
                    import pyautogui
                    pyautogui.press('right')
                    logger.info("Fallback keystroke successful")
                    return True
                except Exception as e2:
                    logger.error(f"Fallback keystroke also failed: {e2}")
            
            return False
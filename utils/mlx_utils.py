"""
MLX Utilities for Apple Silicon optimization

This module provides helper functions for working with MLX on Apple Silicon M-series chips.
"""

import os
import logging
import platform

# Configure logging
logger = logging.getLogger("mlx_utils")

# Check for Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine().startswith('arm')
MLX_AVAILABLE = False

# Only attempt to import MLX on Apple Silicon
if IS_APPLE_SILICON:
    try:
        import mlx
        import mlx.core as mx
        MLX_AVAILABLE = True
        logger.info("MLX detected and available for Apple Silicon acceleration")
        
        # Get Apple chip information
        # This can help identify M1/M2/M3/M4 chips 
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                chip_info = result.stdout.strip()
                if "Apple M" in chip_info:
                    logger.info(f"Detected {chip_info}")
                    
                    # Store chip generation (M1, M2, M3, M4)
                    if "M1" in chip_info:
                        APPLE_CHIP = "M1"
                    elif "M2" in chip_info:
                        APPLE_CHIP = "M2" 
                    elif "M3" in chip_info:
                        APPLE_CHIP = "M3"
                    elif "M4" in chip_info:
                        APPLE_CHIP = "M4"
                    else:
                        APPLE_CHIP = "Unknown"
                        
                    logger.info(f"Using optimizations for Apple {APPLE_CHIP}")
        except Exception as e:
            logger.warning(f"Could not determine exact Apple chip model: {e}")
            APPLE_CHIP = "Unknown"
            
    except ImportError as e:
        logger.warning(f"MLX not available on this system: {e}")
        logger.info("Install MLX for optimal performance: pip install mlx")
else:
    logger.info("Not running on Apple Silicon - MLX optimizations will not be used")

def get_optimal_batch_size():
    """
    Return the optimal batch size for the current Apple chip.
    
    Different M-series chips have different optimal batch sizes
    depending on their Neural Engine capabilities.
    """
    if not MLX_AVAILABLE:
        return 1
        
    # Different optimal batch sizes based on chip generation
    if APPLE_CHIP == "M4":
        return 32  # M4 Neural Engine can handle larger batches
    elif APPLE_CHIP == "M3":
        return 16  # M3 Neural Engine
    elif APPLE_CHIP == "M2":
        return 8   # M2 Neural Engine
    elif APPLE_CHIP == "M1":
        return 4   # M1 Neural Engine
    else:
        return 8   # Default for unknown Apple Silicon

def is_mlx_available():
    """Simple function to check if MLX is available."""
    return MLX_AVAILABLE

def get_mlx_version():
    """Get MLX version information."""
    if not MLX_AVAILABLE:
        return "MLX not available"
    
    return mlx.__version__

def get_device_info():
    """Return information about the current device."""
    if IS_APPLE_SILICON:
        try:
            # Get chip details
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                total_ram = int(result.stdout.strip()) / (1024**3)  # Convert to GB
                
            return {
                "device": f"Apple {APPLE_CHIP}",
                "total_ram": f"{total_ram:.1f} GB",
                "mlx_available": MLX_AVAILABLE,
                "mlx_version": get_mlx_version() if MLX_AVAILABLE else None
            }
        except Exception as e:
            logger.warning(f"Error getting device info: {e}")
    
    return {
        "device": platform.machine(),
        "mlx_available": False
    } 
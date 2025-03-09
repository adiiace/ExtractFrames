# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import time
import psutil
import threading
import queue
import argparse
import json
import sys
import re
import glob
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import platform
from tqdm import tqdm
import logging
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()
buffer_size = 0
# Default configuration with comments
DEFAULT_CONFIG = {
    # Core settings
    "ram_usage_percent": 70,  # Percentage of available RAM to use for frame buffering
    "max_buffer_frames": None,  # Maximum frames to buffer, None for auto-calculation
    "use_gpu": True,  # Whether to use GPU acceleration if available
    "jpeg_quality": 95,  # JPEG quality (1-100)
    "worker_threads": None,  # Number of worker threads, None for auto (CPU count)
    "gpu_monitoring": True,  # Whether to monitor GPU usage during processing
    
    # Output settings
    "output_format": "jpg",  # Output format: jpg, png, webp
    "output_template": "frame_{:06d}",  # Template for frame filenames
    "resize_output": None,  # Resize output frames to this size (e.g. "1280x720" or None for original)
    "extract_interval": 1,  # Extract every nth frame (1 = all frames)
    
    # Batch processing
    "recursive_scan": False,  # Whether to scan subdirectories for videos
    "video_extensions": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],  # Supported video extensions
    
    # Preview mode
    "preview_mode": False,  # Whether to extract sample frames before full processing
    "preview_frames": 10,  # Number of frames to extract in preview mode
    "preview_interval": "equal",  # How to select preview frames: "start", "equal", "random"
    
    # Visual settings
    "colors": {
        "success": "green",  # Color for success messages
        "error": "red",  # Color for error messages
        "warning": "yellow",  # Color for warning messages
        "info": "cyan",  # Color for info messages
        "highlight": "magenta",  # Color for highlighted content
        "normal": "white"  # Default text color
    },
    
    # Customizable messages
    "messages": {
        "start": "Starting frame extraction...",
        "complete": "Frame extraction completed successfully!",
        "error_video_open": "Error: Could not open video file. Please check the path.",
        "error_output_dir": "Error: Could not create output directory.",
        "info_gpu_available": "CUDA is available with {devices} device(s)",
        "info_gpu_unavailable": "CUDA is NOT available in OpenCV. Using CPU mode.",
        "info_using_gpu": "Using GPU acceleration for video decoding",
        "info_using_cpu": "Using CPU for video decoding",
        "batch_start": "Starting batch processing of {count} videos...",
        "batch_complete": "Batch processing completed. Processed {success} videos successfully, {failed} failed.",
        "preview_complete": "Preview extraction complete. Sample frames available in {dir}"
    }
}

# Set up color mapping
COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE
}

def colored_print(message, color_name="normal", is_bold=False):
    """Print colored text using colorama"""
    config = load_config()
    color = COLOR_MAP.get(config["colors"].get(color_name, "white"), Fore.WHITE)
    bold = Style.BRIGHT if is_bold else ""
    print(f"{bold}{color}{message}{Style.RESET_ALL}")

def get_config_path():
    """Determine the appropriate config file path based on the platform"""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        base_path = Path(sys.executable).parent
    else:
        # Running as a script
        base_path = Path(__file__).parent
    
    # Try app directory first
    config_path = base_path / "extract_frames_config.json"
    
    # Check if the directory is writable
    if os.access(base_path, os.W_OK):
        return config_path
    
    # If not writable, use user config directory
    if platform.system() == "Windows":
        config_dir = Path(os.environ.get("APPDATA")) / "ExtractFrames"
    else:  # Linux/Mac
        config_dir = Path.home() / ".config" / "ExtractFrames"
    
    # Create directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    return config_dir / "config.json"

def load_config():
    """Load configuration from file or create default if not exists"""
    config_path = get_config_path()
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all fields exist
                merged_config = DEFAULT_CONFIG.copy()
                
                # Recursively update config while preserving nested structures
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                            d[k] = update_dict(d[k].copy(), v)
                        else:
                            d[k] = v
                    return d
                
                merged_config = update_dict(merged_config, config)
                return merged_config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config file: {e}", "error")
            print("Using default configuration", "warning")
            return DEFAULT_CONFIG
    else:
        # Create default config file with comments
        try:
            with open(config_path, 'w') as f:
                # Generate commented config
                config_str = json.dumps(DEFAULT_CONFIG, indent=4)
                f.write(config_str)
            colored_print(f"Created default configuration file at: {config_path}", "info")
        except IOError as e:
            colored_print(f"Warning: Could not write default config file: {e}", "warning")
        
        return DEFAULT_CONFIG

def save_frame(frame_data):
    """Save a single frame to disk"""
    frame_num, frame, output_dir, output_format, quality, output_template, resize = frame_data
    
    # Determine file extension and format
    if output_format.lower() == "png":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)]
    elif output_format.lower() == "webp":
        ext = ".webp"
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:  # default to jpg
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    # Apply template
    base_filename = output_template.format(frame_num)
    output_path = os.path.join(output_dir, f"{base_filename}{ext}")
    
    # Resize if requested
    if resize is not None:
        try:
            width, height = resize
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.warning(f"Failed to resize frame {frame_num}: {e}")
    
    # Save frame
    cv2.imwrite(output_path, frame, params)
    return frame_num

def check_gpu_support(config):
    """Check if OpenCV is built with CUDA support and if CUDA is available"""
    has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if has_cuda:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        colored_print(config["messages"]["info_gpu_available"].format(devices=device_count), "info")
        for i in range(device_count):
            colored_print(f"Device {i}: {cv2.cuda.getDevice()}", "info")
        return True
    else:
        colored_print(config["messages"]["info_gpu_unavailable"], "warning")
        return False

def monitor_gpu_usage():
    """Monitor GPU usage in a separate thread"""
    if platform.system() == "Windows":
        # For Windows systems
        while not gpu_monitor_stop.is_set():
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                usage = output.decode().strip()
                colored_print(f"GPU Utilization: {usage}%", "info")
            except:
                colored_print("Could not monitor GPU (nvidia-smi not available)", "warning")
                break
            time.sleep(5)
    else:
        # For Linux systems
        while not gpu_monitor_stop.is_set():
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                usage = output.decode().strip()
                colored_print(f"GPU Utilization: {usage}%", "info")
            except:
                colored_print("Could not monitor GPU (nvidia-smi not available)", "warning")
                break
            time.sleep(5)

def get_video_duration(cap):
    """Get the duration of a video in seconds"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 0

def format_time(seconds):
    """Format seconds as a time string"""
    return str(timedelta(seconds=int(seconds)))

def extract_preview_frames(video_path, output_dir, config):
    """Extract a small number of preview frames from the video"""
    preview_frames = config["preview_frames"]
    preview_interval = config["preview_interval"]
    
    # Create preview subfolder
    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        colored_print("Could not open video for preview", "error")
        return False
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        colored_print("Could not determine video length", "error")
        cap.release()
        return False
    
    # Determine which frames to extract
    frame_indices = []
    if preview_interval == "start":
        # First N frames
        frame_indices = list(range(min(preview_frames, total_frames)))
    elif preview_interval == "random":
        # Random frames
        import random
        frame_indices = sorted(random.sample(range(total_frames), min(preview_frames, total_frames)))
    else:  # "equal" or any other value
        # Evenly spaced frames
        if total_frames < preview_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / preview_frames
            frame_indices = [int(i * step) for i in range(preview_frames)]
    
    # Extract frames
    quality = config["jpeg_quality"]
    output_format = config["output_format"]
    output_template = config["output_template"]
    resize_output = config["resize_output"]
    
    # Parse resize dimensions
    resize = None
    if resize_output:
        match = re.match(r"(\d+)x(\d+)", resize_output)
        if match:
            resize = (int(match.group(1)), int(match.group(2)))
    
    # Create progress bar
    pbar = tqdm(total=len(frame_indices), desc="Extracting preview frames", 
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Style.RESET_ALL))
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            save_frame((i, frame, preview_dir, output_format, quality, output_template, resize))
            pbar.update(1)
    
    cap.release()
    pbar.close()
    
    colored_print(config["messages"]["preview_complete"].format(dir=preview_dir), "success")
    return True

def extract_frames(video_path, output_dir, config):
    """Extract frames from video with configuration settings"""
    # Get configuration values
    ram_usage_percent = config.get("ram_usage_percent", 70)
    max_buffer_frames = config.get("max_buffer_frames")
    use_gpu = config.get("use_gpu", True)
    jpeg_quality = config.get("jpeg_quality", 95)
    worker_threads = config.get("worker_threads")
    gpu_monitoring = config.get("gpu_monitoring", True)
    output_format = config.get("output_format", "jpg")
    output_template = config.get("output_template", "frame_{:06d}")
    extract_interval = config.get("extract_interval", 1)
    preview_mode = config.get("preview_mode", False)
    resize_output = config.get("resize_output")
    
    # Parse resize dimensions
    resize = None
    if resize_output:
        match = re.match(r"(\d+)x(\d+)", resize_output)
        if match:
            resize = (int(match.group(1)), int(match.group(2)))
    
    # Print start message
    colored_print(config["messages"]["start"], "highlight", True)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        colored_print(config["messages"]["error_output_dir"], "error")
        return 1
    
    # Run preview mode if enabled
    if preview_mode:
        extract_preview_frames(video_path, output_dir, config)
    
    # Check GPU support
    has_cuda = check_gpu_support(config) if use_gpu else False
    
    # Set up GPU monitoring if requested and GPU is available
    global gpu_monitor_stop
    gpu_monitor_stop = threading.Event()
    gpu_monitor_thread = None
    
    if gpu_monitoring and has_cuda:
        gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage)
        gpu_monitor_thread.daemon = True
        gpu_monitor_thread.start()
    
    # Enable CUDA acceleration if available
    if has_cuda:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|hwaccel;cuda|hwaccel_output_format;cuda"
        colored_print(config["messages"]["info_using_gpu"], "info")
    else:
        colored_print(config["messages"]["info_using_cpu"], "info")
    
    # Open video
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 128)  # Initial buffer size
    
    if not cap.isOpened():
        colored_print(config["messages"]["error_video_open"], "error")
        return 1
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = get_video_duration(cap)
    
    colored_print(f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames, {format_time(duration)}", "highlight")
    
    # Calculate effective frames with interval
    effective_frames = total_frames // extract_interval
    
    # Calculate frame size and set memory limits (in frames)
    frame_size_bytes = width * height * 3  # RGB frame size
    available_ram_bytes = psutil.virtual_memory().available * (ram_usage_percent / 100.0)
    max_frames_in_ram = int(available_ram_bytes / frame_size_bytes)
    
    # Use user-provided buffer size or calculate based on RAM
    if max_buffer_frames is not None:
        buffer_size = max_buffer_frames
    else:
        buffer_size = min(max_frames_in_ram, 1000)  # Default to 1000 max to avoid excessive memory use
    
    colored_print(f"Frame size: {frame_size_bytes/1024/1024:.2f} MB", "info")
    colored_print(f"Available RAM for buffering: {available_ram_bytes/1024/1024/1024:.2f} GB", "info")
    colored_print(f"Using buffer size of {buffer_size} frames", "info")
    
    # Determine number of worker threads
    if worker_threads is None:
        worker_threads = max(os.cpu_count(), 1)
    colored_print(f"Using {worker_threads} worker threads for frame saving", "info")
    
    # Create producer-consumer queues
    frame_queue = queue.Queue(maxsize=buffer_size)
    done_event = threading.Event()
    
    # Progress bars with colors
    read_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)
    save_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL)
    
    read_pbar = tqdm(total=total_frames, desc="Reading frames", position=0, 
                     bar_format=read_format)
    save_pbar = tqdm(total=effective_frames, desc="Saving frames", position=1, 
                     bar_format=save_format)
    
    # Producer thread - reads frames from video
    def frame_producer():
        global buffer_size
        frame_count = 0
        saved_frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on interval
            if frame_count % extract_interval == 0:
                # Convert frame to use CUDA if possible
                if has_cuda:
                    try:
                        # Try to use VRAM by keeping frame in GPU memory
                        cuda_frame = cv2.cuda_GpuMat()
                        cuda_frame.upload(frame)
                        processed_frame = cuda_frame.download()  # Download when needed
                    except cv2.error:
                        # Fall back to CPU if CUDA operations fail
                        processed_frame = frame
                        if saved_frame_count == 0:
                            colored_print("WARNING: CUDA upload failed, falling back to CPU processing", "warning")
                else:
                    processed_frame = frame
                    
                # Wait if queue is full (this implements backpressure)
                while frame_queue.full() and not done_event.is_set():
                    time.sleep(0.01)
                    
                # Put frame in queue for saving
                frame_queue.put((saved_frame_count, processed_frame, output_dir, 
                                output_format, jpeg_quality, output_template, resize))
                saved_frame_count += 1
            
            frame_count += 1
            read_pbar.update(1)
            
            # Update stats less frequently to reduce console spam
            if frame_count % 100 == 0:
                if not read_pbar.disable:
                    read_pbar.set_postfix({"Queue": f"{frame_queue.qsize()}/{buffer_size}"})
                    
                    # Dynamic buffer resizing based on memory pressure
                    if psutil.virtual_memory().percent > 95 and buffer_size > 100:
                        new_buffer_size = max(100, buffer_size // 2)
                        colored_print(f"WARNING: High memory usage detected! Reducing buffer from {buffer_size} to {new_buffer_size} frames", "warning")
                        # We can't resize the queue, but we can limit how many items we put in it
                        buffer_size = new_buffer_size
        
        cap.release()
        done_event.set()
        read_pbar.close()
    
    # Consumer function - saves frames to disk
    def frame_consumer():
        saved_count = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=worker_threads) as executor:
            future_to_frame = {}
            
            while True:
                try:
                    # Get frame from queue with timeout
                    frame_data = frame_queue.get(timeout=1.0)
                    
                    # Submit frame for saving
                    future = executor.submit(save_frame, frame_data)
                    future_to_frame[future] = frame_data[0]
                    
                    # Process completed futures
                    completed = [f for f in future_to_frame if f.done()]
                    for future in completed:
                        try:
                            frame_num = future_to_frame[future]
                            future.result()  # Check for exceptions
                            future_to_frame.pop(future)
                            saved_count += 1
                            save_pbar.update(1)
                        except Exception as e:
                            colored_print(f"Error saving frame: {e}", "error")
                        
                        # Update statistics less frequently
                        if saved_count % 100 == 0 and not save_pbar.disable:
                            elapsed = time.time() - start_time
                            fps = saved_count / elapsed if elapsed > 0 else 0
                            ram_usage = psutil.virtual_memory().percent
                            save_pbar.set_postfix({"FPS": f"{fps:.1f}", "RAM": f"{ram_usage}%"})
                    
                    # Check if we're done
                    if done_event.is_set() and frame_queue.empty() and not future_to_frame:
                        break
                        
                except queue.Empty:
                    if done_event.is_set():
                        # Process remaining futures
                        for future in list(future_to_frame.keys()):
                            try:
                                future.result()  # Wait for completion
                                save_pbar.update(1)
                            except Exception as e:
                                colored_print(f"Error in remaining frames: {e}", "error")
                            future_to_frame.pop(future)
                        
                        if not future_to_frame:
                            break
        
        elapsed = time.time() - start_time
        save_pbar.close()
        fps_rate = saved_count/elapsed if elapsed > 0 else 0
        colored_print(f"Saved {saved_count} frames in {elapsed:.2f} seconds ({fps_rate:.2f} fps)", "info")
    
    # Start processing
    start_time = time.time()
    
    # Start producer thread
    producer_thread = threading.Thread(target=frame_producer)
    producer_thread.start()
    
    # Start consumer process
    consumer_thread = threading.Thread(target=frame_consumer)
    consumer_thread.start()
    
    # Wait for completion
    producer_thread.join()
    consumer_thread.join()
    
    # Stop GPU monitoring if it was started
    if gpu_monitor_thread is not None:
        gpu_monitor_stop.set()
        gpu_monitor_thread.join(timeout=1.0)
    
    total_time = time.time() - start_time
    colored_print(f"Total processing time: {total_time:.2f} seconds", "info")
    colored_print(config["messages"]["complete"], "success", True)
    
    return 0  # Success return code

def process_directory(input_path, output_base, config):
    """Process a directory of video files"""
    extensions = config["video_extensions"]
    recursive = config["recursive_scan"]
    
    # Build file pattern
    file_patterns = [f"*{ext}" for ext in extensions]
    
    # Collect all video files
    all_videos = []
    if recursive:
        for pattern in file_patterns:
            all_videos.extend(glob.glob(os.path.join(input_path, "**", pattern), recursive=True))
    else:
        for pattern in file_patterns:
            all_videos.extend(glob.glob(os.path.join(input_path, pattern)))
    
    # Sort files
    all_videos.sort()
    
    if not all_videos:
        colored_print("No video files found in the specified directory", "warning")
        return 1
    
    colored_print(config["messages"]["batch_start"].format(count=len(all_videos)), "highlight", True)
    
    # Process each video
    success_count = 0
    failed_count = 0
    
    for i, video_path in enumerate(all_videos):
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Create output directory
        video_output_dir = os.path.join(output_base, f"{video_name}_frames")
        
        # Print separator
        colored_print(f"\n[{i+1}/{len(all_videos)}] Processing {video_filename}", "highlight", True)
        
        # Process the video
        try:
            result = extract_frames(video_path, video_output_dir, config)
            if result == 0:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            colored_print(f"Error processing {video_filename}: {e}", "error")
            failed_count += 1
    
    # Print batch summary
    colored_print(config["messages"]["batch_complete"].format(
        success=success_count, failed=failed_count), 
        "success" if failed_count == 0 else "warning", True)
    
    return 0 if failed_count == 0 else 1

def main():
    # Load configuration
    config = load_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract frames from a video with GPU acceleration')
    parser.add_argument('input_path', type=str, help='Path to the input video file or directory')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, 
                        help='Directory to save extracted frames (optional, default: auto-created)')
    parser.add_argument('--ram-usage', type=int, default=config["ram_usage_percent"], 
                        help=f'Percentage of available RAM to use for frame buffering (default: {config["ram_usage_percent"]})')
    parser.add_argument('--max-buffer', type=int, default=config["max_buffer_frames"], 
                        help='Maximum number of frames to buffer in RAM (default: auto)')
    parser.add_argument('--cpu-only', action='store_true', 
                        help='Disable GPU acceleration and use CPU only')
    parser.add_argument('--quality', type=int, default=config["jpeg_quality"], 
                        help=f'JPEG quality (1-100, default: {config["jpeg_quality"]})')
    parser.add_argument('--threads', type=int, default=config["worker_threads"], 
                        help='Number of worker threads for saving frames (default: number of CPU cores)')
    parser.add_argument('--no-gpu-monitor', action='store_true', 
                        help='Disable GPU usage monitoring')
    parser.add_argument('--format', type=str, default=config["output_format"],
                        choices=['jpg', 'png', 'webp'], 
                        help=f'Output format (default: {config["output_format"]})')
    parser.add_argument('--interval', type=int, default=config["extract_interval"],
                        help=f'Extract every nth frame (default: {config["extract_interval"]})')
    parser.add_argument('--resize', type=str, default=config["resize_output"],
                        help='Resize output frames (e.g. "1280x720", default: original size)')
    parser.add_argument('--preview', action='store_true',
                        help='Extract sample frames before full processing')
    parser.add_argument('--recursive', action='store_true',
                        help='Scan subdirectories when processing a directory')
    parser.add_argument('--save-config', action='store_true',
                        help='Save current command line options as new defaults in config file')
    
    args = parser.parse_args()
    
    # Update config if requested
    if args.save_config:
        new_config = config.copy()
        new_config["ram_usage_percent"] = args.ram_usage
        new_config["max_buffer_frames"] = args.max_buffer
        new_config["use_gpu"] = not args.cpu_only
        new_config["jpeg_quality"] = args.quality
        new_config["worker_threads"] = args.threads
        new_config["gpu_monitoring"] = not args.no_gpu_monitor
        new_config["output_format"] = args.format
        new_config["extract_interval"] = args.interval
        new_config["resize_output"] = args.resize
        new_config["preview_mode"] = args.preview
        new_config["recursive_scan"] = args.recursive
        
        try:
            with open(get_config_path(), 'w') as f:
                json.dump(new_config, f, indent=4)
            colored_print(f"Updated configuration saved to: {get_config_path()}", "success")
        except IOError as e:
            colored_print(f"Error saving configuration: {e}", "error")
    
    # Override config with command line args
    run_config = config.copy()
    run_config["ram_usage_percent"] = args.ram_usage
    run_config["max_buffer_frames"] = args.max_buffer
    run_config["use_gpu"] = not args.cpu_only
    run_config["jpeg_quality"] = args.quality
    run_config["worker_threads"] = args.threads
    run_config["gpu_monitoring"] = not args.no_gpu_monitor
    run_config["output_format"] = args.format
    run_config["extract_interval"] = args.interval
    run_config["resize_output"] = args.resize
    run_config["preview_mode"] = args.preview
    run_config["recursive_scan"] = args.recursive
    
    # Check input path type
    input_path = args.input_path
    
    # Auto-create output directory if not specified
    if args.output_dir is None:
        input_name = os.path.basename(os.path.splitext(input_path)[0])
        if os.path.isdir(input_path):
            output_dir = os.path.join(os.path.dirname(input_path), f"{input_name}")
        else:
            output_dir = os.path.join(os.path.dirname(input_path), f"{input_name}_frames")
    else:
        output_dir = args.output_dir
    
    # Process based on whether input is a directory or file
    if os.path.isdir(input_path):
        return process_directory(input_path, output_dir, run_config)
    elif os.path.isfile(input_path):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        return extract_frames(input_path, output_dir, run_config)
    else:
        colored_print(f"Error: Input path '{input_path}' does not exist", "error")
        return 1

def get_system_info():
    """Get system information for troubleshooting"""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "cuda_support": cv2.cuda.getCudaEnabledDeviceCount() > 0
    }
    
    # Get GPU info if available
    if info["cuda_support"]:
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
            gpu_info = output.decode().strip().split(',')
            info["gpu_name"] = gpu_info[0].strip()
            info["gpu_memory_gb"] = float(gpu_info[1].strip()) / 1024
            info["gpu_driver"] = gpu_info[2].strip()
        except:
            info["gpu_name"] = "Unknown (nvidia-smi not available)"
            info["gpu_memory_gb"] = 0
            info["gpu_driver"] = "Unknown"
    
    return info

def display_system_info():
    """Display system information in a formatted way"""
    info = get_system_info()
    
    colored_print("\n===== System Information =====", "highlight", True)
    colored_print(f"Platform: {info['platform']}", "info")
    colored_print(f"Python: {info['python']}", "info")
    colored_print(f"OpenCV: {info['opencv']}", "info")
    colored_print(f"CPU Cores: {info['cpu_count']}", "info")
    colored_print(f"Total RAM: {info['ram_total_gb']:.2f} GB", "info")
    
    if info["cuda_support"]:
        colored_print("\n----- GPU Information -----", "highlight")
        colored_print(f"GPU: {info['gpu_name']}", "info")
        colored_print(f"GPU Memory: {info['gpu_memory_gb']:.2f} GB", "info")
        colored_print(f"GPU Driver: {info['gpu_driver']}", "info")
    else:
        colored_print("\nNo CUDA-compatible GPU detected", "warning")
    
    colored_print("=============================\n", "highlight", True)

def print_banner():
    """Print application banner"""
    banner = r"""
    ┌─────────────────────────────────────────┐
    │  ExtractFrames - Advanced Video Frame   │
    │       Extraction Utility v2.0.0         │
    └─────────────────────────────────────────┘
    """
    colored_print(banner, "highlight", True)

def check_updates():
    """Check for updates to the application"""
    # just a placeholder
    current_version = "2.0.0"
    try:
        # Simulated check - in a real app, this would make an HTTP request
        latest_version = current_version  # Placeholder
        
        # else:
        #     colored_print("You have the latest version!", "success")
    except:
        pass  # Silent fail for update checks

def estimate_output_size(video_path, config):
    """Estimate the output size of the extraction process"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Adjust for interval
    effective_frames = frame_count // config["extract_interval"]
    
    # Estimate size based on format
    bytes_per_pixel = 3  # RGB
    raw_frame_size = width * height * bytes_per_pixel
    
    # Apply format-specific compression estimates
    if config["output_format"].lower() == "png":
        # PNG compression varies widely, estimate conservatively
        compression_ratio = 0.5  # About 50% of raw size
    elif config["output_format"].lower() == "webp":
        # WebP is very efficient
        compression_ratio = 0.25  # About 25% of raw size
    else:  # JPG
        # JPEG compression depends on quality
        quality = config["jpeg_quality"]
        if quality >= 95:
            compression_ratio = 0.4
        elif quality >= 90:
            compression_ratio = 0.3
        elif quality >= 80:
            compression_ratio = 0.2
        else:
            compression_ratio = 0.1
    
    # Calculate estimated size
    estimated_size_bytes = raw_frame_size * effective_frames * compression_ratio
    estimated_size_gb = estimated_size_bytes / (1024**3)
    
    # Apply resize factor if specified
    if config["resize_output"]:
        try:
            match = re.match(r"(\d+)x(\d+)", config["resize_output"])
            if match:
                new_width, new_height = int(match.group(1)), int(match.group(2))
                resize_factor = (new_width * new_height) / (width * height)
                estimated_size_gb *= resize_factor
        except:
            pass  # If resize parsing fails, use original estimate
    
    return estimated_size_gb

def setup_logging():
    """Set up logging to file"""
    log_dir = Path.home() / ".logs" / "extractframes"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = log_dir / f"extract_frames_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Redirect warnings to logging
    logging.captureWarnings(True)
    
    return log_file

def auto_optimize_config(video_path, config):
    """Automatically optimize configuration based on video and system properties"""
    optimized_config = config.copy()
    
    # Check if the video is 4K or higher
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Adjust RAM usage based on video resolution
        if width * height >= 3840 * 2160:  # 4K
            colored_print("Detected 4K or higher resolution video", "info")
            # For 4K, reduce buffer size to avoid excessive memory usage
            optimized_config["max_buffer_frames"] = min(optimized_config.get("max_buffer_frames", 1000), 500)
            
            # Suggest lower quality for large videos
            if frame_count > 10000 and optimized_config["jpeg_quality"] > 90:
                optimized_config["jpeg_quality"] = 90
                colored_print("Automatically reduced JPEG quality to 90 for large 4K video", "info")
        
        # Check if the video is high framerate (>60fps)
        if fps > 60:
            colored_print(f"Detected high framerate video ({fps} fps)", "info")
            # For high fps videos, suggest extracting fewer frames
            if optimized_config["extract_interval"] == 1:
                suggested_interval = max(1, round(fps / 30))  # Target around 30fps output
                colored_print(f"Consider using --interval={suggested_interval} to extract fewer frames", "info")
    
    # Check system memory
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    if total_ram_gb < 8:  # Less than 8GB RAM
        colored_print("Detected low system memory, reducing buffer size", "warning")
        optimized_config["max_buffer_frames"] = min(optimized_config.get("max_buffer_frames", 500), 250)
        optimized_config["ram_usage_percent"] = min(optimized_config["ram_usage_percent"], 50)
    
    # Check number of CPU cores
    if os.cpu_count() <= 2:  # Dual-core or single-core system
        colored_print("Detected limited CPU resources, reducing thread count", "warning")
        optimized_config["worker_threads"] = 1
    
    return optimized_config

def error_handler(func):
    """Decorator for handling and logging errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            colored_print("\nOperation cancelled by user", "warning")
            return 130  # SIGINT return code
        except Exception as e:
            colored_print(f"Error: {str(e)}", "error")
            logging.exception("Unhandled exception:")
            return 1
    return wrapper

@error_handler
def run_cli():
    """Main entry point with error handling"""
    print_banner()
    log_file = setup_logging()
    check_updates()
    
    # Log system info
    logging.info("Starting ExtractFrames")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        logging.info(f"System info - {key}: {value}")
    
    result = main()
    
    if result == 0:
        colored_print(f"\nAll operations completed successfully!", "success", True)
        colored_print(f"Log file saved to: {log_file}", "info")
    else:
        colored_print(f"\nOperation completed with errors. See log file for details: {log_file}", "warning", True)
    
    return result

if __name__ == "__main__":
    sys.exit(run_cli())

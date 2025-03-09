# ExtractFramesV2 

## Overview
ExtractFramesV2 is a Python script for extracting frames from video files with support for GPU acceleration (CUDA) and multi-threading. It provides various options for frame extraction, including interval-based extraction, resizing, batch processing, and preview mode. The script is optimized for performance, allowing customization of RAM usage, buffer size, and output quality.

## Features
- **GPU Acceleration:** Uses CUDA if available for faster frame extraction.
- **Multi-threading:** Utilizes multiple worker threads for efficient frame saving.
- **Configurable Settings:** Adjustable RAM usage, frame interval, output format, and resizing.
- **Batch Processing:** Process multiple video files in a directory.
- **Preview Mode:** Extracts a small number of frames for quick review before full processing.
- **Custom Logging & Messages:** Provides real-time progress updates and customizable messages.
- **Supports Multiple Formats:** Extract frames in JPG, PNG, or WebP formats.

## Requirements
- Python 3.7+
- OpenCV (with CUDA support for GPU acceleration)
- Additional Dependencies:
  ```bash
  pip install opencv-python tqdm colorama psutil numpy
  ```
- (Optional) NVIDIA GPU with CUDA for hardware acceleration

## Installation
1. Clone or download the script.
2. Install dependencies using the command:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure OpenCV is compiled with CUDA if using GPU acceleration.

## Usage
### Basic Frame Extraction
```bash
python ExtractFramesV2.py input_video.mp4 output_frames/
```

### Extract Every Nth Frame
```bash
python ExtractFramesV2.py input_video.mp4 output_frames/ --interval 5
```

### Enable/Disable GPU Acceleration
```bash
python ExtractFramesV2.py input_video.mp4 output_frames/ --cpu-only
```

### Resize Extracted Frames
```bash
python ExtractFramesV2.py input_video.mp4 output_frames/ --resize 1280x720
```

### Batch Processing (Process All Videos in a Directory)
```bash
python ExtractFramesV2.py input_videos/ output_frames/ --recursive
```

### Save Custom Settings as Default
```bash
python ExtractFramesV2.py input_video.mp4 output_frames/ --save-config
```

## Configuration
The script automatically loads settings from `extract_frames_config.json`. If the file does not exist, it creates one with default values. Users can modify this file to change settings permanently.

## Logging & Monitoring
- GPU and RAM usage can be monitored during processing.
- Logs are stored in the user's config directory.

## Troubleshooting
- Ensure OpenCV is installed with CUDA support if GPU acceleration is required.
- Check available system memory and adjust buffer size accordingly.
- Run with `--cpu-only` if experiencing issues with GPU processing.

## License
This project is licensed under the MIT License.


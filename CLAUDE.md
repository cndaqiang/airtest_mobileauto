# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirTest Mobile Automation is an object-oriented, multi-process control framework for mobile apps based on the AirTest framework. It enhances AirTest with stability features including connection checks, automatic retries on failure, and automatic reconnection for continuous operation. Primary use case is automating mobile games (particularly Honor of Kings) on Android/iOS devices.

## Core Architecture

### Main Module: `airtest_mobileauto/control.py` (~2150 lines)

This is the primary module containing all core functionality. Key classes and their responsibilities:

#### 1. **Settings** (line 120)
- Global configuration class managing all runtime parameters
- Uses `Config(config_file)` to load settings from YAML files
- Key configuration areas:
  - Node info: `mynode`, `totalnode`, `multiprocessing` for multi-process support
  - Device connection: `LINK_dict`, `BlueStackdir`, `LDPlayerdir`, `MuMudir`, `dockercontain`
  - Time management: Uses UTC+8 (GMT+08:00) timezone via `eastern_eight_tz` for Chinese game cycles
  - Logging: `logger_level`, `logfile_dict` for formatted output with `[MM-DD HH:MM:SS]` timestamps
  - Platform detection: Automatically determines Windows/Linux/macOS control environment

#### 2. **DQWheel** (line 817) - "Tool" utilities
- Multi-process coordination via file system synchronization
- Time management: `timelimit(timekey, limit, init, reset)` for throttling operations
- File operations: `removefile()`, `removefiles()`, synchronization barriers
- Dictionary/position caching: Stores image coordinates in YAML files to reduce repeated template matching
- Temporary file management in `Settings.tmpdir`

#### 3. **deviceOB** (line 1520) - Device Management
- Object-oriented device connection and lifecycle management
- Supported control endpoints:
  - **Windows**: BlueStacks, LDPlayer, MuMu emulators (with start/stop/restart)
  - **Linux**: Docker containers (redroid)
  - **macOS**: iOS devices via tidevice
  - **Cross-platform**: USB Android, WiFi Android, custom commands
- Auto-detects client type from LINK string and Settings configuration
- Key methods:
  - `连接设备(times, timesMax)`: Retry connection logic with device restart on final attempt
  - `重启重连设备()`: Full device restart and reconnection
  - Client detection logic (lines 1550-1568): Determines if BlueStacks/LD/MuMu/Docker/USB/Remote based on Settings and platform

#### 4. **appOB** (line 1939) - APP Management
- App lifecycle: `打开APP()`, `关闭APP()`, `重启APP(sleeptime)`
- APPID validation and correction for regional variations (e.g., mark.via vs mark.via.gp)
- `前台APP()`: Get foreground app (Android only)
- Handles Activity specification: Supports both "com.example.app" and "com.example.app/Activity" formats

#### 5. **TaskManager** (line 2092) - Multi-process Execution
- Orchestrates single vs multi-process task execution
- Uses `multiprocessing.Pool` for parallel execution across `totalnode` processes
- Each process re-reads config and sets node-specific parameters via `Config_mynode()`

### Enhanced AirTest Functions (lines 612-750)

The framework wraps core AirTest functions to add resilience:
- `connect_device()`: Retry wrapper around `connect_device_o()`
- `exists()`, `touch()`, `swipe()`: Auto-retry with connection status checking
- `start_app()`: Special handling for Android monkey errors with `--pct-syskeys 0` parameter
- All wrappers check `connect_status()` and retry once on failure before raising error

### Utility Module: `airtest_mobileauto/pick2yaml.py`

Simple script for migrating dictionary files from `.txt` to `.yaml` format and removing spaces from keys.

### OCR Module: `airtest_mobileauto/ocr.py` (Optional)

Lightweight OCR (Optical Character Recognition) module for text recognition in images, complementing AirTest's image-based recognition. **This is an optional feature** that requires extra dependencies.

**Key Components:**

- **OCREngine class**: EasyOCR (PyTorch) wrapper providing:
  - `recognize_text(img)`: Recognize all text in image, returns list of results with positions and confidence
  - `find_text(img, target_text)`: Find specific text, supports exact/fuzzy matching
  - `find_all_text(img, target_text)`: Find all occurrences of text
  - Auto GPU/CPU detection and fallback

- **Coordinate utilities**:
  - `bbox_to_center()`: Convert bounding box to center point
  - `abs_to_relative()`: Convert absolute pixel coordinates to relative (0-1 range)
  - `relative_to_abs()`: Convert relative to absolute coordinates

- **Result format**: All recognition functions return dictionaries with:
  ```python
  {
      'text': str,                    # Recognized text
      'confidence': float,            # Confidence (0-1)
      'bbox': (x, y, w, h),          # Absolute pixel coordinates
      'center': (cx, cy),            # Center point
      'relative_bbox': (rx, ry, rw, rh),  # Relative coordinates
      'relative_center': (rcx, rcy)       # Relative center
  }
  ```

**Design Rationale:**
- Uses EasyOCR (PyTorch) for best user experience
- CUDA libraries bundled in PyTorch wheel - no system CUDA installation needed
- GPU acceleration ready out-of-the-box, auto-fallback to CPU
- ~1.5GB deployment size acceptable since OCR is optional and not bundled
- Supports 80+ languages including Chinese and English
- Optional dependency: Install with `pip install airtest_mobileauto[ocr]`
- Gracefully degrades when not installed - core library remains lightweight

## Build and Development Commands

### Installation

**Standard installation (lightweight):**
```bash
pip install airtest_mobileauto
```

**With OCR support:**
```bash
pip install airtest_mobileauto[ocr]
```

**With all optional features:**
```bash
pip install airtest_mobileauto[all]  # Includes OCR + Windows features
```

**No-dependencies install (if dependencies already installed):**
```bash
pip install airtest_mobileauto --no-deps
```

**For Python 3.7 (Windows):**
```powershell
.\localbuild37.ps1
```

### Building Distribution

**Build source distribution:**
```bash
rm ./dist/*
python setup.py sdist
python -m pip install ./dist/airtest_mobileauto-<version>.tar.gz
```

**Upload to PyPI:**
```bash
twine upload dist/*
```

### Nuitka Compilation (Python 3.7 Windows 32-bit)

Compile core modules to `.pyd` for code protection:
```powershell
.\build_airtest_nuitka.ps1
```

This script:
1. Installs package normally
2. Compiles `control.py` and `pick2yaml.py` to `.pyd` using Nuitka with mingw64
3. Removes source `.py` files from site-packages
4. Copies `.pyd` files to site-packages

The `.pyd` files are platform/version-specific (e.g., `control.cp37-win32.pyd`).

## Configuration

### YAML Config Structure

Projects using this framework typically use YAML config files with structure:
- `mynode`: Current process ID (0-based)
- `totalnode`: Total number of parallel processes
- `multiprocessing`: Enable/disable multi-process mode
- `prefix`: Unique identifier for temporary file isolation
- `tmpdir`: Directory for inter-process synchronization files
- `figdir`: Directory containing template images (default: "assets")
- LINK_dict: Dictionary mapping node IDs to device connection strings
- Device-specific settings: `BlueStackdir`, `LDPlayerdir`, `MuMudir`, `dockercontain`, `win_Instance`

See [configuration guide](https://cndaqiang.github.io/wzry.doc/guide/config/) for detailed format.

## Key Design Patterns

### Multi-process Coordination
- File-based synchronization barriers using temporary files in `tmpdir`
- Each process has unique node ID (`mynode`) from 0 to `totalnode-1`
- Synchronization files: `.tmp.barrier.*.txt` in `Settings.tmpdir`

### Resilience Strategy
- All AirTest operations wrapped with retry logic
- Connection status checked before retry attempts
- Device restart triggered after max connection failures
- Screen update flag (`Settings.screen_update`) tracks when UI state may have changed

### Time Zone Handling
- All time operations use `Settings.eastern_eight_tz` (UTC+8)
- Critical for game automation with daily reset cycles at specific times
- Use `timenow()` function instead of `datetime.now()` for timezone-aware timestamps

### Template Matching Optimization
- Image coordinates cached in YAML dictionaries via DQWheel
- `var_dict_file` stores positions to avoid re-matching identical templates
- Supports position selection (e.g., "least proficient hero" in game automation)

## Testing

No formal test suite in repository. Testing approach:
- Development examples: [autowzry](https://github.com/cndaqiang/autowzry), [autoansign](https://github.com/MobileAutoFlow/autoansign)
- Verify imports after installation:
```bash
python -c "import airtest_mobileauto; print(airtest_mobileauto.__file__)"
python -c "from airtest_mobileauto import control; print('success')"
```

## Platform-Specific Notes

### Windows
- Emulator control via COM automation or executable commands
- Boss key support for hiding apps (configurable key combinations in `BossKeydict`)
- Requires `pywin32` for Windows-specific features

### Linux
- Docker container management via Docker CLI commands
- Redroid containers supported for Android emulation

### macOS
- iOS device control via `tidevice`
- Physical device reconnection required if `tidevice list` doesn't show device

## Dependencies

### Core Dependencies (Always Required)
- **airtest==1.3.6**: Core framework for device automation
- **pyyaml**: Configuration file parsing and position caching
- **opencv-contrib-python>=4.4.0.46,<=4.6.0.66**: Image processing
- **numpy<2.0**: Used by airtest for image matching
- **pywin32** (Windows only, auto-installed): Windows-specific emulator control and Boss key functionality

### Optional Dependencies

**OCR Features (`pip install airtest_mobileauto[ocr]`):**
- **easyocr>=1.7.0**: PyTorch-based OCR engine
  - CUDA libraries bundled in PyTorch wheel (no system CUDA installation needed)
  - Adds ~1.5GB to deployment size
  - GPU acceleration ready out-of-the-box (auto-fallback to CPU if no GPU)
  - Cross-platform compatible (Windows/Linux/macOS/ARM)
  - Chinese and English text recognition with 80+ languages support

### OCR Module Usage

The OCR module gracefully handles missing dependencies:
```python
from airtest_mobileauto.ocr import OCREngine

try:
    ocr = OCREngine()  # Will raise ImportError if not installed
    result = ocr.find_text(img, '确定')
except ImportError as e:
    print(e)  # Provides installation instructions
    # Core library functions continue to work normally
```

See `OCR_README.md` for detailed OCR usage documentation.

## Important Files

- `tpl_target_pos.png`: Test template image for position validation (included in package data)
- `control.py`: Main automation logic (can be compiled to .pyd)
- `pick2yaml.py`: Migration utility (can be compiled to .pyd)
- `.build/` directories: Nuitka compilation artifacts (can be cleaned up)

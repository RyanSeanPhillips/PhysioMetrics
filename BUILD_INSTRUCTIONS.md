# PhysioMetrics - Windows Executable Build Instructions

This document provides step-by-step instructions for packaging the PhysioMetrics breath analysis application into a standalone Windows executable.

## Prerequisites

### Required Software
- **Python 3.8+** with pip
- **Windows 10/11** (for testing the executable)
- **Git** (for version control)

### Required Python Packages
Install the required packages using the provided requirements file:
```bash
pip install -r requirements.txt
```

## Quick Start

### Method 1: Using the Batch Script (Easiest)
1. Open Command Prompt in the project directory
2. Run the build script:
   ```cmd
   build_executable.bat
   ```
3. Wait for the build to complete (5-10 minutes)
4. Find your executable in `dist/PhysioMetrics.exe`

### Method 2: Using the Python Script (Recommended)
1. Run the Python build script:
   ```bash
   python build_executable.py
   ```
   This provides better error checking and cross-platform compatibility.

### Method 3: Manual PyInstaller (Advanced)
1. Generate the version info file:
   ```bash
   python version_info.py
   ```
2. Run PyInstaller with the spec file:
   ```bash
   pyinstaller --clean physiometrics.spec
   ```

## File Structure

```
physiometrics/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── physiometrics.spec            # PyInstaller configuration
├── build_executable.bat      # Windows build script
├── build_executable.py       # Cross-platform build script
├── version_info.py          # Version metadata generator
├── run_debug.py             # Debug launcher for testing
├── core/                    # Core application modules
├── ui/                      # UI definition files
├── images/                  # Application icons
└── assets/                  # Additional assets
```

## Testing the Application

### Before Building
1. Test the application directly:
   ```bash
   python run_debug.py
   ```
2. Verify all features work correctly
3. Check that all data files load properly

### After Building
1. Test the executable on the build machine
2. **IMPORTANT**: Test on a clean Windows machine without Python installed
3. Verify all features work in the executable version
4. Test with sample data files

## Troubleshooting

### Common Build Issues

#### "Module not found" errors
- **Solution**: Add missing modules to `hidden_imports` in `physiometrics.spec`
- **Example**: If scipy.signal is missing, add `'scipy.signal'` to the list

#### UI files not found
- **Solution**: Verify UI files exist in the `ui/` directory
- **Check**: The `datas` section in `physiometrics.spec` includes UI files

#### Large executable size
- **Expected**: 200-400 MB due to scientific libraries (NumPy, SciPy, Matplotlib)
- **To reduce**: Add more modules to the `excludes` list in the spec file

#### Slow startup time
- **Expected**: 3-5 seconds on first launch due to library initialization
- **Alternative**: Use directory distribution instead of single file (see spec file comments)

### Runtime Issues

#### Application crashes on startup
1. Run the executable from Command Prompt to see error messages:
   ```cmd
   dist\PhysioMetrics.exe
   ```
2. Check for missing data files or dependencies

#### Features not working
1. Verify all data files are included in the `datas` section
2. Test the source code version to isolate the issue

## Distribution

### Single File Distribution
- **File**: `dist/PhysioMetrics.exe`
- **Size**: ~200-400 MB
- **Pros**: Easy to distribute, single file
- **Cons**: Slower startup, larger download

### Directory Distribution
To create a directory distribution (faster startup):
1. Uncomment the COLLECT section in `physiometrics.spec`
2. Comment out the single-file EXE section
3. Rebuild the application
4. Distribute the entire `dist/PhysioMetrics/` folder

### Creating an Installer (Optional)
For professional distribution, consider creating an installer using:
- **NSIS** (Nullsoft Scriptable Install System)
- **Inno Setup**
- **WiX Toolset**

## Version Management

### Updating Version Information
1. Edit `version_info.py` to update version numbers
2. Rebuild the executable
3. The version info will be embedded in the .exe properties

### Release Checklist
- [ ] Update version number in `version_info.py`
- [ ] Test all application features
- [ ] Build executable
- [ ] Test executable on clean machine
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Distribute to users

## Performance Optimization

### Reducing Build Time
- Use `--clean` flag only when necessary
- Remove unused imports from the codebase
- Cache PyInstaller builds when possible

### Reducing Executable Size
- Add unused modules to `excludes` in the spec file
- Remove unnecessary data files
- Use UPX compression (enabled by default)

### Improving Startup Time
- Use directory distribution instead of single file
- Pre-compile Python bytecode
- Optimize import statements

## Security Considerations

### Code Signing (Recommended for Distribution)
Consider signing the executable to avoid Windows security warnings:
1. Obtain a code signing certificate
2. Use `signtool.exe` to sign the executable
3. This prevents "Unknown Publisher" warnings

### Antivirus False Positives
PyInstaller executables may trigger antivirus software:
- This is common with PyInstaller-generated executables
- Code signing helps reduce false positives
- Submit the executable to antivirus vendors if needed

## Support and Updates

### Updating the Application
To update the application:
1. Make changes to the source code
2. Test thoroughly
3. Rebuild the executable
4. Distribute the new version

### User Support
- Provide users with sample data files for testing
- Include this documentation with distributions
- Consider creating a simple user manual

## Advanced Configuration

### Custom Icons
- Replace icons in the `images/` directory
- Update the icon path in `physiometrics.spec`
- Rebuild the executable

### Adding Modules
To add new Python modules:
1. Add to `requirements.txt`
2. Add to `hidden_imports` in `physiometrics.spec` if needed
3. Test and rebuild

### Platform Support
This configuration is optimized for Windows. For other platforms:
- Modify the spec file for macOS/Linux
- Adjust file paths and icon formats
- Test on target platforms

---

For questions or issues, refer to the [PyInstaller documentation](https://pyinstaller.readthedocs.io/) or create an issue in the project repository.
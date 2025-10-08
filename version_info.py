"""
Version information for PlethApp Windows executable
This file contains metadata that will be embedded in the .exe file
"""

# Version information
VERSION = (1, 0, 4, 0)
VERSION_STRING = "1.0.4"

# Windows version info structure
version_info = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={VERSION},
    prodvers={VERSION},
    mask=0x3f,
    flags=0x0,
    OS=0x4,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'Breath Analysis Lab'),
            StringStruct(u'FileDescription', u'PlethApp - Advanced Breath Analysis Tool'),
            StringStruct(u'FileVersion', u'{VERSION_STRING}'),
            StringStruct(u'InternalName', u'PlethApp'),
            StringStruct(u'LegalCopyright', u'Copyright © 2024 Ryan Phillips'),
            StringStruct(u'OriginalFilename', u'PlethApp.exe'),
            StringStruct(u'ProductName', u'PlethApp Breath Analysis'),
            StringStruct(u'ProductVersion', u'{VERSION_STRING}'),
            StringStruct(u'Comments', u'Advanced respiratory signal analysis with eupnea and apnea detection')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""

# Save version info to file
def create_version_file():
    """Create the version_info.txt file for PyInstaller."""
    with open('version_info.txt', 'w') as f:
        f.write(version_info)
    print("Created version_info.txt")

if __name__ == '__main__':
    create_version_file()
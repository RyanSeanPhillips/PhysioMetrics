"""
Simple version bumper for PlethApp
Run this before committing to increment the patch version
"""
import re

def bump_patch_version():
    """Increment the patch version (e.g., 1.0.2 -> 1.0.3)"""
    with open('version_info.py', 'r') as f:
        content = f.read()

    # Find current version
    version_match = re.search(r'VERSION = \((\d+), (\d+), (\d+), (\d+)\)', content)
    string_match = re.search(r'VERSION_STRING = "(\d+)\.(\d+)\.(\d+)"', content)

    if not version_match or not string_match:
        print("Error: Could not find version information")
        return

    major, minor, patch, build = map(int, version_match.groups())
    patch += 1

    # Update version tuple
    new_version = f"VERSION = ({major}, {minor}, {patch}, {build})"
    content = re.sub(r'VERSION = \(\d+, \d+, \d+, \d+\)', new_version, content)

    # Update version string
    new_string = f'VERSION_STRING = "{major}.{minor}.{patch}"'
    content = re.sub(r'VERSION_STRING = "\d+\.\d+\.\d+"', new_string, content)

    # Write back
    with open('version_info.py', 'w') as f:
        f.write(content)

    print(f"Version bumped to {major}.{minor}.{patch}")

if __name__ == '__main__':
    bump_patch_version()

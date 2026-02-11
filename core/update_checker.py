"""
GitHub Release Update Checker

Checks for new PhysioMetrics releases on GitHub and notifies users.
Supports both stable releases and beta/prerelease versions.
"""

import urllib.request
import json
import re
from typing import Optional, Tuple, List
from version_info import VERSION_STRING


# GitHub repository info
GITHUB_REPO_OWNER = "RyanSeanPhillips"
GITHUB_REPO_NAME = "PhysioMetrics"
GITHUB_API_URL_LATEST = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
GITHUB_API_URL_ALL = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases"


def is_prerelease_version(version_str: str) -> bool:
    """
    Check if a version string indicates a prerelease (beta, alpha, rc, dev).

    Args:
        version_str: Version string like "1.0.15-beta.2" or "1.0.14"

    Returns:
        True if this is a prerelease version
    """
    version_lower = version_str.lower()
    prerelease_markers = ['beta', 'alpha', 'rc', 'dev', 'preview', 'pre']
    return any(marker in version_lower for marker in prerelease_markers)


def parse_version(version_str: str) -> Tuple[Tuple[int, ...], str]:
    """
    Parse semantic version string into tuple of integers and prerelease suffix.

    Args:
        version_str: Version string like "1.0.11", "v1.0.11", or "1.0.15-beta.2"

    Returns:
        Tuple of (version_tuple, prerelease_suffix)
        e.g., ((1, 0, 15), "beta.2") or ((1, 0, 14), "")
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip('v')

    # Split on dash to separate version from prerelease suffix
    parts = version_str.split('-', 1)
    main_version = parts[0]
    prerelease = parts[1] if len(parts) > 1 else ""

    # Parse main version
    try:
        version_tuple = tuple(int(x) for x in main_version.split('.'))
    except (ValueError, AttributeError):
        version_tuple = (0, 0, 0)

    return version_tuple, prerelease


def parse_prerelease_number(prerelease: str) -> int:
    """
    Parse prerelease suffix to extract version number for comparison.

    Args:
        prerelease: Prerelease suffix like "beta.2" or "rc1"

    Returns:
        Integer for comparison (higher = newer)
    """
    if not prerelease:
        return 1000000  # Stable releases are "higher" than any prerelease

    # Extract numbers from prerelease string
    numbers = re.findall(r'\d+', prerelease)
    if numbers:
        return int(numbers[-1])  # Use last number (e.g., "beta.2" -> 2)
    return 0


def compare_versions(current: str, latest: str) -> int:
    """
    Compare two version strings.

    Args:
        current: Current version string
        latest: Latest version string

    Returns:
        1 if latest > current
        0 if latest == current
        -1 if latest < current
    """
    current_tuple, current_pre = parse_version(current)
    latest_tuple, latest_pre = parse_version(latest)

    # Compare main version first
    if latest_tuple > current_tuple:
        return 1
    elif latest_tuple < current_tuple:
        return -1

    # Same main version - compare prerelease
    # Note: stable (no prerelease) > beta > alpha
    current_pre_num = parse_prerelease_number(current_pre)
    latest_pre_num = parse_prerelease_number(latest_pre)

    if latest_pre_num > current_pre_num:
        return 1
    elif latest_pre_num < current_pre_num:
        return -1

    return 0


def fetch_all_releases(timeout: float = 5.0) -> List[dict]:
    """
    Fetch all releases from GitHub API.

    Args:
        timeout: Request timeout in seconds

    Returns:
        List of release dictionaries from GitHub API
    """
    try:
        req = urllib.request.Request(GITHUB_API_URL_ALL)
        req.add_header('Accept', 'application/vnd.github.v3+json')

        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"[Update Check] Error fetching releases: {e}")
        return []


def check_for_updates(timeout: float = 5.0) -> Optional[dict]:
    """
    Check GitHub for new releases, handling both stable and beta versions.

    For stable versions: checks if there's a newer stable release.
    For beta versions: checks for newer beta releases AND shows latest stable.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary with update info:
        {
            'is_beta': bool,  # Is current version a beta?
            'stable_update': {...} or None,  # Newer stable available?
            'beta_update': {...} or None,  # Newer beta available? (only if running beta)
            'latest_stable': {...},  # Info about latest stable (for reference)
        }
        Returns None on error.
    """
    try:
        releases = fetch_all_releases(timeout)
        if not releases:
            return None

        # Separate stable and prerelease versions
        stable_releases = [r for r in releases if not r.get('prerelease', False) and not r.get('draft', False)]
        prerelease_releases = [r for r in releases if r.get('prerelease', False) and not r.get('draft', False)]

        # Get latest stable
        latest_stable = stable_releases[0] if stable_releases else None
        latest_prerelease = prerelease_releases[0] if prerelease_releases else None

        is_beta = is_prerelease_version(VERSION_STRING)

        result = {
            'is_beta': is_beta,
            'stable_update': None,
            'beta_update': None,
            'latest_stable': None,
            'latest_beta': None,
        }

        # Store latest stable info
        if latest_stable:
            latest_stable_version = latest_stable.get('tag_name', '').lstrip('v')
            result['latest_stable'] = {
                'version': latest_stable_version,
                'url': latest_stable.get('html_url', GITHUB_RELEASES_URL),
                'name': latest_stable.get('name', f'Version {latest_stable_version}'),
                'published_at': latest_stable.get('published_at', ''),
                'body': latest_stable.get('body', '')
            }

            # Check if there's a newer stable version
            if compare_versions(VERSION_STRING, latest_stable_version) > 0:
                result['stable_update'] = result['latest_stable']

        # Store latest beta info (if running beta)
        if is_beta and latest_prerelease:
            latest_pre_version = latest_prerelease.get('tag_name', '').lstrip('v')
            result['latest_beta'] = {
                'version': latest_pre_version,
                'url': latest_prerelease.get('html_url', GITHUB_RELEASES_URL),
                'name': latest_prerelease.get('name', f'Beta {latest_pre_version}'),
                'published_at': latest_prerelease.get('published_at', ''),
                'body': latest_prerelease.get('body', '')
            }

            # Check if there's a newer beta version
            if compare_versions(VERSION_STRING, latest_pre_version) > 0:
                result['beta_update'] = result['latest_beta']

        return result

    except Exception as e:
        print(f"[Update Check] Error checking for updates: {e}")
        return None


def get_update_message(update_info: dict) -> str:
    """
    Format update info into user-friendly message for About tab.

    Handles both stable and beta version scenarios:
    - If running stable: shows stable update if available
    - If running beta: shows beta status + latest stable + beta update if available

    Args:
        update_info: Dictionary from check_for_updates()

    Returns:
        Formatted HTML message with box styling
    """
    if update_info is None:
        return get_check_failed_message()

    is_beta = update_info.get('is_beta', False)
    stable_update = update_info.get('stable_update')
    beta_update = update_info.get('beta_update')
    latest_stable = update_info.get('latest_stable', {})

    messages = []

    # If running beta, show beta status first
    if is_beta:
        stable_version = latest_stable.get('version', 'Unknown') if latest_stable else 'Unknown'
        messages.append(f"""
        <div style="padding: 12px; background-color: #3a3a4a; border: 2px solid #7B68EE; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #9B89EE; margin-top: 0;"> Beta Version</h3>
            <p style="margin: 8px 0;">You are running <b>v{VERSION_STRING}</b> (beta/prerelease)</p>
            <p style="margin: 8px 0;">Latest stable version: <b>v{stable_version}</b></p>
            <p style="margin-bottom: 0; font-size: 11px; color: #888;">
                Beta versions may contain bugs. Report issues on GitHub.
            </p>
        </div>
        """)

    # Show beta update if available (running beta and newer beta exists)
    if beta_update:
        version = beta_update.get('version', 'Unknown')
        name = beta_update.get('name', f'Beta {version}')
        messages.append(f"""
        <div style="padding: 15px; background-color: #3a4a3a; border: 3px solid #7B68EE; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #9B89EE; margin-top: 0;"> Newer Beta Available!</h3>
            <p style="margin: 8px 0;"><b>New beta:</b> v{version}</p>
            <p style="margin: 8px 0;"><b>Release:</b> {name}</p>
            <p style="margin-bottom: 0;">
                <a href="{beta_update.get('url', GITHUB_RELEASES_URL)}" style="color: #9B89EE; font-weight: bold; text-decoration: underline;">
                    Download latest beta →
                </a>
            </p>
        </div>
        """)

    # Show stable update if available (and not running beta, or stable is newer than beta's base)
    if stable_update and not is_beta:
        version = stable_update.get('version', 'Unknown')
        name = stable_update.get('name', f'Version {version}')
        messages.append(f"""
        <div style="padding: 15px; background-color: #2a4a2a; border: 3px solid #4CAF50; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #4CAF50; margin-top: 0;"> Update Available!</h3>
            <p style="margin: 8px 0;"><b>New version:</b> v{version}</p>
            <p style="margin: 8px 0;"><b>Release:</b> {name}</p>
            <p style="margin-bottom: 0;">
                <a href="{stable_update.get('url', GITHUB_RELEASES_URL)}" style="color: #4CAF50; font-weight: bold; text-decoration: underline;">
                    Download the latest version →
                </a>
            </p>
        </div>
        """)

    # If no updates and not beta, show "up to date"
    if not messages:
        messages.append(f"""
        <div style="padding: 10px; background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 5px;">
            <p style="color: #4CAF50; margin: 5px 0;"><b>✓ You're up to date!</b></p>
            <p style="margin: 5px 0;">Current version: v{VERSION_STRING}</p>
        </div>
        """)

    return '\n'.join(messages)


def get_main_window_update_message(update_info: dict) -> Optional[Tuple[str, str]]:
    """
    Get simple update message for main window banner.

    Args:
        update_info: Dictionary from check_for_updates()

    Returns:
        Tuple of (text, url) for display in main window, or None if no update
    """
    if update_info is None:
        return None

    is_beta = update_info.get('is_beta', False)
    beta_update = update_info.get('beta_update')
    stable_update = update_info.get('stable_update')

    # For beta users, prioritize beta updates
    if is_beta and beta_update:
        version = beta_update.get('version', 'Unknown')
        url = beta_update.get('url', GITHUB_RELEASES_URL)
        return (f"New Beta Available: v{version}", url)

    # For stable users, show stable updates
    if stable_update and not is_beta:
        version = stable_update.get('version', 'Unknown')
        url = stable_update.get('url', GITHUB_RELEASES_URL)
        return (f"Update Available: v{version}", url)

    return None


def get_no_update_message() -> str:
    """Get message when no update is available."""
    is_beta = is_prerelease_version(VERSION_STRING)

    if is_beta:
        return f"""
        <div style="padding: 10px; background-color: #3a3a4a; border: 1px solid #5a5a6a; border-radius: 5px;">
            <p style="color: #9B89EE; margin: 5px 0;"><b> Running Beta Version</b></p>
            <p style="margin: 5px 0;">Current version: v{VERSION_STRING}</p>
            <p style="margin: 5px 0; color: #888; font-size: 11px;">
                You're running the latest beta. Thanks for testing!
            </p>
        </div>
        """
    else:
        return f"""
        <div style="padding: 10px; background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 5px;">
            <p style="color: #4CAF50; margin: 5px 0;"><b>✓ You're up to date!</b></p>
            <p style="margin: 5px 0;">Current version: v{VERSION_STRING}</p>
        </div>
        """


def get_check_failed_message() -> str:
    """Get message when update check fails."""
    is_beta = is_prerelease_version(VERSION_STRING)
    beta_note = " (beta)" if is_beta else ""

    return f"""
    <div style="padding: 10px; background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 5px;">
        <p style="margin: 5px 0;">Current version: v{VERSION_STRING}{beta_note}</p>
        <p style="margin: 5px 0; color: #888;">
            <i>Could not check for updates (network error)</i>
        </p>
        <p style="margin: 5px 0;">
            <a href="{GITHUB_RELEASES_URL}" style="color: #2a7fff;">
                View releases on GitHub →
            </a>
        </p>
    </div>
    """

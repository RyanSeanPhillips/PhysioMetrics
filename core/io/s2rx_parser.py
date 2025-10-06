"""
Parser for Spike2 .s2rx configuration files.

This module extracts channel visibility and display settings from Spike2's
XML configuration files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set, Optional


def parse_s2rx_visibility(s2rx_path: str) -> Dict[int, bool]:
    """
    Parse a .s2rx file and extract channel visibility settings.
    
    Args:
        s2rx_path: Path to .s2rx configuration file
        
    Returns:
        Dictionary mapping channel numbers to visibility (True=visible, False=hidden)
        Returns empty dict if file doesn't exist or can't be parsed
    """
    s2rx_file = Path(s2rx_path)
    
    if not s2rx_file.exists():
        return {}
    
    try:
        tree = ET.parse(s2rx_file)
        root = tree.getroot()
        
        # Find all <Chan> elements with visibility info
        visibility = {}
        
        for chan_elem in root.findall('.//View/Chan'):
            chan_attr = chan_elem.get('Chan')
            vis_attr = chan_elem.get('Vis')
            
            if chan_attr is None:
                continue
            
            # Parse channel number (may be simple like "4" or complex like "800,5a")
            # For now, just extract the first integer
            try:
                # Handle both simple ("4") and complex ("800,5a") channel identifiers
                if ',' in chan_attr:
                    # This is a virtual/processed channel - skip for now
                    continue
                
                chan_num = int(chan_attr)
                
                # Vis attribute: "0" = hidden, absent or "1" = visible
                is_visible = (vis_attr != "0")
                
                visibility[chan_num] = is_visible
                
            except (ValueError, TypeError):
                # Skip invalid channel numbers
                continue
        
        return visibility
        
    except Exception as e:
        # If parsing fails, return empty dict (show all channels)
        print(f"Warning: Failed to parse {s2rx_path}: {e}")
        return {}


def get_hidden_channels(smrx_path: str) -> Optional[Set[int]]:
    """
    Get the set of HIDDEN channel numbers for a .smrx file.

    Looks for a .s2rx file with the same base name and parses visibility.
    This returns HIDDEN channels because channels not mentioned in .s2rx
    default to VISIBLE.

    Args:
        smrx_path: Path to .smrx data file

    Returns:
        Set of hidden channel numbers, or None if no .s2rx file found
        (None means no filtering - show all channels)
    """
    smrx_file = Path(smrx_path)
    s2rx_file = smrx_file.with_suffix('.s2rx')

    if not s2rx_file.exists():
        # No config file - show all channels
        return None

    visibility = parse_s2rx_visibility(str(s2rx_file))

    if not visibility:
        # Couldn't parse or empty - show all channels
        return None

    # Return set of channels explicitly marked as HIDDEN (Vis="0")
    # Channels not mentioned in the file default to visible
    hidden = {ch_num for ch_num, is_vis in visibility.items() if not is_vis}

    return hidden


# Example usage
if __name__ == "__main__":
    test_file = r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\100225_000.smrx"

    hidden_channels = get_hidden_channels(test_file)

    if hidden_channels is None:
        print("No .s2rx file found - showing all channels")
    else:
        print(f"Hidden channels from .s2rx: {sorted(hidden_channels)}")

    # Also test direct parsing
    s2rx_file = Path(test_file).with_suffix('.s2rx')
    if s2rx_file.exists():
        visibility = parse_s2rx_visibility(str(s2rx_file))
        print(f"\nFull visibility map (only explicitly set channels):")
        for ch_num in sorted(visibility.keys()):
            status = "VISIBLE" if visibility[ch_num] else "HIDDEN"
            print(f"  Channel {ch_num}: {status}")
        print("\nNote: Channels not listed above default to VISIBLE")

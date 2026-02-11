"""
Plot Theme Manager for PhysioMetrics

Provides consistent theming for matplotlib plots across the application.
Currently used by Advanced Editor dialog, designed for future integration
into main plotting window.
"""

from typing import Dict


# Dark Theme - Optimized for long viewing sessions
DARK_THEME = {
    # Background
    'figure_facecolor': '#000000',  # True black for sharp appearance
    'axes_facecolor': '#000000',    # True black for plot area

    # Grid and borders
    'grid_color': '#3e3e42',
    'grid_alpha': 0.3,
    'axes_edgecolor': '#3e3e42',

    # Text and labels
    'text_color': '#d4d4d4',
    'label_color': '#cccccc',
    'tick_color': '#cccccc',

    # Signal trace
    'trace_color': '#d4d4d4',       # Off-white (softer than pure white, matches text color)
    'trace_linewidth': 0.8,

    # Peak markers (inspiratory peaks)
    'peak_color': '#ff0000',         # Red (matches main plot PEAK_COLOR)
    'peak_marker': '^',
    'peak_size': 8,

    # Disagreement highlight
    'disagreement_peak_color': '#ff6b6b',  # Bright red/pink for visibility
    'disagreement_peak_marker': 'v',
    'disagreement_peak_size': 12,
    'disagreement_span_color': '#ff6b6b50',  # Translucent

    # Event markers (match main plot exactly)
    'onset_color': '#2ecc71',        # Green (inspiratory onset)
    'offset_color': '#f39c12',       # Orange (inspiratory offset)
    'expmin_color': '#1f78b4',       # Blue (expiratory minimum)
    'expoff_color': '#9b59b6',       # Purple (expiratory offset)

    # Model prediction overlays
    'model_a_overlay': '#00800040',  # Green translucent
    'model_b_overlay': '#80000040',  # Red translucent

    # Sniff regions
    'sniff_color': '#ffff0030',      # Yellow translucent
    'eupnea_color': '#00ff0030',     # Green translucent
}


# Light Theme - For presentations and publications
LIGHT_THEME = {
    # Background
    'figure_facecolor': '#ffffff',
    'axes_facecolor': '#f9f9f9',

    # Grid and borders
    'grid_color': '#cccccc',
    'grid_alpha': 0.5,
    'axes_edgecolor': '#888888',

    # Text and labels
    'text_color': '#000000',
    'label_color': '#333333',
    'tick_color': '#333333',

    # Signal trace
    'trace_color': '#000000',        # Black (good for light theme)
    'trace_linewidth': 0.8,

    # Peak markers (inspiratory peaks)
    'peak_color': '#cc0000',         # Dark red
    'peak_marker': '^',
    'peak_size': 8,

    # Disagreement highlight
    'disagreement_peak_color': '#ff0000',  # Bright red
    'disagreement_peak_marker': 'v',
    'disagreement_peak_size': 12,
    'disagreement_span_color': '#ff000030',

    # Event markers (match main plot colors, darker for light bg)
    'onset_color': '#27ae60',        # Darker green
    'offset_color': '#e67e22',       # Darker orange
    'expmin_color': '#1f78b4',       # Blue (same as dark theme)
    'expoff_color': '#8e44ad',       # Darker purple

    # Model prediction overlays
    'model_a_overlay': '#00cc0050',
    'model_b_overlay': '#cc000050',

    # Sniff regions
    'sniff_color': '#ffcc0050',
    'eupnea_color': '#00cc0050',
}


class PlotThemeManager:
    """Manages plot themes for matplotlib figures."""

    def __init__(self, default_theme='dark'):
        """
        Initialize theme manager.

        Args:
            default_theme: 'dark' or 'light'
        """
        self.current_theme = default_theme
        self.themes = {
            'dark': DARK_THEME,
            'light': LIGHT_THEME,
        }

    def apply_theme(self, ax, fig, theme_name=None):
        """
        Apply theme to matplotlib figure and axes.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object
            theme_name: 'dark' or 'light' (if None, uses current_theme)
        """
        if theme_name is None:
            theme_name = self.current_theme
        else:
            self.current_theme = theme_name

        theme = self.themes[theme_name]

        # Figure background
        fig.patch.set_facecolor(theme['figure_facecolor'])

        # Axes background
        ax.set_facecolor(theme['axes_facecolor'])

        # Spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(theme['axes_edgecolor'])

        # Grid
        ax.grid(True, alpha=theme['grid_alpha'], color=theme['grid_color'])

        # Tick colors
        ax.tick_params(colors=theme['tick_color'], which='both')

        # Label colors
        ax.xaxis.label.set_color(theme['label_color'])
        ax.yaxis.label.set_color(theme['label_color'])

        # Title color
        if ax.get_title():
            ax.title.set_color(theme['text_color'])

    def get_color(self, element_name):
        """
        Get color for a specific plot element.

        Args:
            element_name: Name of element (e.g., 'trace_color', 'peak_color')

        Returns:
            Color string (hex or rgba)
        """
        return self.themes[self.current_theme].get(element_name, '#ffffff')

    def get_value(self, key):
        """
        Get any theme value by key.

        Args:
            key: Theme parameter key

        Returns:
            Theme value
        """
        return self.themes[self.current_theme].get(key)

    def switch_theme(self, theme_name):
        """
        Switch to a different theme.

        Args:
            theme_name: 'dark' or 'light'
        """
        if theme_name in self.themes:
            self.current_theme = theme_name

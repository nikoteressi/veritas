"""
Matplotlib configuration to avoid font warnings and improve performance.
This module should be imported early in the application startup.
"""

import logging
import warnings

import matplotlib

logger = logging.getLogger(__name__)


def configure_matplotlib():
    """Configure matplotlib to avoid font warnings and improve startup performance."""
    try:
        # Use Agg backend (non-interactive) to avoid GUI dependencies
        matplotlib.use("Agg")

        # Configure matplotlib to avoid font cache rebuilding
        import matplotlib.pyplot as plt

        # Set font family to avoid problematic system fonts
        plt.rcParams["font.family"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]

        # Disable font warnings
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

        # Suppress specific matplotlib warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        warnings.filterwarnings("ignore", message=".*font.*", module="matplotlib")

        logger.info("✅ Matplotlib configured successfully")

    except Exception as e:
        logger.warning(f"⚠️ Failed to configure matplotlib: {e}")


def suppress_font_warnings():
    """Suppress matplotlib font-related warnings."""
    # Suppress font manager warnings
    warnings.filterwarnings("ignore", message=".*Failed to extract font properties.*")
    warnings.filterwarnings("ignore", message=".*Can not load face.*")
    warnings.filterwarnings("ignore", message=".*Could not set the fontsize.*")

    # Set matplotlib font manager logging to ERROR level
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# Apply configuration when module is imported
suppress_font_warnings()

"""Data-only lettering library lifecycle, safe for project worker threads."""
from modules.utils.lettering_styles import empty_style_library, normalize_style_library


def reset_lettering_project(main) -> None:
    main.lettering_styles = empty_style_library()


def styles_for_project(main, library=None) -> dict:
    return normalize_style_library(getattr(main, 'lettering_styles', None) if library is None else library)

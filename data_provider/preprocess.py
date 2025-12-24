
"""
Thin wrapper for backward compatibility. Delegates to cli.main.
"""

from cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

"""Open Vision entry point. Run directly: python __main__.py

The Windows console is forced to UTF-8 inside cli.app, which both this shim and
the `openvision` console script import.
"""
from cli.app import app

def main():
    app()

if __name__ == "__main__":
    main()

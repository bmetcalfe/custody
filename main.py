"""
Custody — entry point shim.

The dashboard is a Streamlit app; there is no meaningful CLI.
Run this file to see usage instructions.
"""
import sys


def main():
    print("Custody - anomaly-aware target custody and collection-planning prototype")
    print()
    print("Launch the dashboard:")
    print("  uv run streamlit run src/app/streamlit_app.py")
    print()
    print("Run the test suite:")
    print("  uv run pytest tests/")


if __name__ == "__main__":
    main()
    sys.exit(0)

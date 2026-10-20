"""
prepare_data.py — Dataset download, inspection, and split creation.

Usage:
    python scripts/prepare_data.py

This script will:
1. Verify the dataset exists in data/raw/.
2. Scan all images, detect corrupt files, count per class.
3. Check for near-duplicate images.
4. Create stratified train/val/test splits.
5. Save split manifests to data/splits/.
6. Save class-to-index mapping.

Will be fully implemented as first real code in Phase 1.
"""


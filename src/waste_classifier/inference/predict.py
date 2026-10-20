"""
predict.py — Reusable single-image inference.

Responsibilities:
- Load saved checkpoint (state_dict + class mapping).
- Accept an image path or PIL Image.
- Apply correct preprocessing.
- Return: predicted class, confidence score, top-3 predictions.

Usage (will be):
    python -m waste_classifier.inference.predict --image path/to/image.jpg

Will be implemented in Phase 5.
"""


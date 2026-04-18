"""
Named Entity Recognition (NER) inference script for extracting animal names.

This script loads a trained spaCy model and identifies entities labeled as "ANIMAL"
within a given text input.
"""
import argparse
import logging
import os

import spacy

logger = logging.getLogger(__name__)


def _validate_text(text) -> None:
    """Raise ValueError if text is not a non-empty string."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")


def extract_from_nlp(text, nlp) -> list[str]:
    """Extract ANIMAL entities using a pre-loaded spaCy pipeline.

    Intended for use with a pipeline object that has already been loaded
    once (e.g. at web-app startup) to avoid repeated disk reads.

    Args:
        text: Input text to analyse.
        nlp: A loaded spaCy Language object.

    Returns:
        Lowercased list of entity texts labelled ANIMAL.
    """
    _validate_text(text)
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]


def extract_animals(text, model_path) -> list[str]:
    """Function to extract animals from text using ONLY the custom model."""
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    _validate_text(text)

    nlp = spacy.load(model_path)
    return extract_from_nlp(text, nlp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Input text for NER")
    parser.add_argument("--model_path", type=str, default="models/ner/custom_ner_model")
    args = parser.parse_args()

    animals_found = extract_animals(args.text, args.model_path)
    logger.info(f"Found animals: {animals_found}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    main()

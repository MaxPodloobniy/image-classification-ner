"""
Training script for custom Named Entity Recognition (NER) model using spaCy.

This module implements a training pipeline for a custom NER model to recognize
the 'ANIMAL' entity type. The script uses spaCy's en_core_web_md base model,
adds a custom NER component, trains it on provided data, and evaluates
performance on a test set.
"""
import spacy
import random
import os
import json
import argparse
import logging
from spacy.training.example import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding

logger = logging.getLogger(__name__)


def parse_args():
    """Function for parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--model_path', type=str, default='models/ner/custom_ner_model')
    parser.add_argument('--train_data', type=str, default='data/ner/train.json')
    parser.add_argument('--test_data', type=str, default='data/ner/test.json')
    return parser.parse_args()


def load_data(filepath):
    """Function for loading json format dataset for model training and testing"""
    logger.info(f"Loading data from {filepath}")
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    data = []
    for item in loaded_data:
        entities = [(start, end, label) for start, end, label in item['entities']]
        data.append((item['text'], {"entities": entities}))

    logger.info(f"Loaded {len(data)} examples from {filepath}")
    return data


def load_or_download_model(model_name: str):
    """
    Load a spaCy model if available.
    If not, print an instruction to install it manually.
    """
    try:
        logger.info(f"Attempting to load model '{model_name}'")
        nlp = spacy.load(model_name)
        logger.info(f"Model '{model_name}' loaded successfully")
        return nlp
    except OSError:
        message = (
            f"\n[ERROR] spaCy model '{model_name}' not found.\n"
            f"Please install it manually by running:\n\n"
            f"    python -m spacy download {model_name}\n"
        )
        logger.error(message)
        raise RuntimeError(f"spaCy model '{model_name}' is missing. Install it and rerun the program.")


def main():
    # Loading all arguments
    args = parse_args()
    logger.info("Starting NER training script")
    logger.info(f"Arguments: {vars(args)}")

    # Load the base model
    nlp = load_or_download_model("en_core_web_md")

    # Add NER component if not already present
    if "ner" not in nlp.pipe_names:
        logger.info("Adding NER component to pipeline")
        ner = nlp.add_pipe("ner", last=True)
    else:
        logger.info("NER component already exists in pipeline")
        ner = nlp.get_pipe("ner")

    # Add custom label "ANIMAL"
    logger.info("Adding custom label 'ANIMAL' to NER")
    ner.add_label("ANIMAL")

    # Load training dataset
    train_dataset = load_data(args.train_data)
    test_dataset = load_data(args.test_data)

    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(f"Test dataset length: {len(test_dataset)}")

    # Prepare training examples for initialization
    logger.info("Preparing training examples")
    train_examples = []
    for text, annotations in train_dataset:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)

    # Disable other pipelines to prevent unnecessary retraining
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    logger.info(f"Disabling pipes: {other_pipes}")

    with nlp.disable_pipes(*other_pipes):
        logger.info("Initializing optimizer")
        optimizer = nlp.initialize(get_examples=lambda: train_examples)

        # Training loop
        logger.info(f"Starting training for {args.epochs} epochs with dropout={args.dropout}")
        for i in range(args.epochs):
            random.shuffle(train_examples)
            losses = {}

            # Update in batches for better stability
            batches = minibatch(train_examples, size=compounding(2.0, 8.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=args.dropout, losses=losses, sgd=optimizer)

            logger.info(f"Epoch {i + 1}/{args.epochs}, Loss: {losses['ner']:.4f}")

        # Save the model
        logger.info(f"Saving model to {args.model_path}")
        os.makedirs(args.model_path, exist_ok=True)
        nlp.to_disk(args.model_path)
        logger.info(f"Model saved to {os.path.abspath(args.model_path)}")

        # Verify the saved model
        try:
            nlp_loaded = spacy.load(args.model_path)
            logger.info("Successfully loaded saved model for verification")
        except Exception as e:
            logger.error(f"Error loading saved model: {e}")

    # Testing saved model
    logger.info("Evaluating model on test data")
    examples = []
    for text, annotations in test_dataset:
        doc = nlp(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    scorer = Scorer()
    scores = scorer.score(examples)

    logger.info(f"Evaluation results:")
    logger.info(f"Precision: {scores['ents_p']:.2%}")
    logger.info(f"Recall: {scores['ents_r']:.2%}")
    logger.info(f"F1-Score: {scores['ents_f']:.2%}")
    logger.info("Training completed successfully")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    main()
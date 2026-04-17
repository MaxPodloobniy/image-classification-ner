# Animal Classification & Verification Pipeline

This project implements an end-to-end pipeline for verifying statements about animals by combining
computer vision and natural language processing. The system analyzes both an image and a text
statement to determine whether the animal mentioned in the text matches the animal shown in the
image.

## Project Overview

The pipeline consists of two main components working together to provide accurate verification.
First, an image classification model identifies what animal is present in a given photograph.
Second, a Named Entity Recognition (NER) model extracts animal names mentioned in a text statement.
Finally, the pipeline compares these results to determine if the statement accurately describes
the image content.

This approach enables automated fact-checking for animal-related claims, which could be useful
in educational contexts, content moderation, wildlife documentation, or any scenario where visual
and textual information need to be verified against each other.

## Architecture

The project combines two deep learning models trained on custom datasets. The image classifier
uses a fine-tuned VGG16 architecture that achieved 94.97% validation accuracy on a custom dataset
of animal images collected from Pinterest and manual sources. The NER model is built on spaCy's
en_core_web_md base model with a custom "ANIMAL" entity type, achieving 92.86% precision, 97.50%
recall, and 95.12% F1-score on the test set.

Both models were trained independently and then integrated into a unified pipeline that processes
multimodal inputs. The verification logic is case-insensitive and can handle multiple animal
mentions in the text, checking if any of the extracted animals match the classified image.

## Project Structure

The repository is organized into distinct directories for each component, along with pipeline
integration and demonstration files:

```
.
├── classifier/                     # Image classification component
│   ├── train_classifier.py         # Training script
│   ├── infer_classifier.py         # Inference script
│   ├── eval_dataset/               # Sample images for testing
│   ├── EDA.ipynb/                  # Exploratory Data Analysis
│   ├── model/                      # Trained model files
│   ├── web-scraper.py              # Script for image scraping
│   └── README.md                   # Detailed classifier documentation
│
├── ner/                            # Named Entity Recognition component
│   ├── train_ner.py                # Training script
│   ├── infer_ner.py                # Inference script
│   ├── data_for_ner/               # Training and test datasets
│   ├── EDA.ipynb/                  # Exploratory Data Analysis
│   ├── model/                      # Trained model files
│   └── README.md                   # Detailed NER documentation
│
├── pipeline.py                     # Main verification pipeline
├── demo.ipynb                      # Jupyter notebook with examples and edge cases
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Installation

To set up the project environment, first clone the repository and install all required
dependencies.
The project requires Python 3.9 along with TensorFlow, spaCy, and other supporting
libraries.

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_md

# (Optional) If you want to use the web scraper for data collection
playwright install
```

All necessary packages and their versions are specified in the `requirements.txt` file, ensuring
reproducible environment setup.

## Usage

### Running the Complete Pipeline

The main pipeline script accepts an image path and a text statement, then outputs whether the
statement is true or false based on the analysis:

```bash
python pipeline.py \
    --image_path path/to/image.jpg \
    --text "I saw a chimpanzee at the zoo." \
    --class_model_path model/classifier_model.keras \
    --ner_model_path custom_ner_model/
```

**Pipeline Arguments:**
- `--image_path` (required) - Path to the image file to analyze
- `--text` (required) - Text statement to verify against the image
- `--class_model_path` (default: model/classifier_model.keras) - Path to the trained image classifier
- `--ner_model_path` (default: custom_ner_model/) - Path to the trained NER model

**Example Output:**

```
2025-10-07 14:23:11 [INFO] Text: I saw a chimpanzee at the zoo.
2025-10-07 14:23:11 [INFO] Extracted animals: chimpanzee
2025-10-07 14:23:12 [INFO] Image classification: chimpanzee (confidence: 98.75%)
2025-10-07 14:23:12 [INFO] ✅ The statement is TRUE!
```

The pipeline returns exit code 0 for true statements and exit code 1 for false statements, making
it easy to integrate into automated workflows or testing scripts.

### Testing Individual Components

Each component can be tested independently using its respective inference script. This is useful
for debugging, evaluating model performance, or integrating individual components into other
projects.

**Image Classification:**
```bash
python classifier/infer_classifier.py \
    --image_path classifier/eval_dataset/chimpanzee.jpg \
    --model_path model/classifier_model.keras
```

**Named Entity Recognition:**
```bash
python ner/infer_ner.py "I saw a chimpanzee and a coyote." \
    --model_path custom_ner_model/
```

### Demo Notebook

The `demo.ipynb` Jupyter notebook provides comprehensive demonstrations of the entire system. It
walks through testing each component individually with various inputs, including edge cases, and
then shows how they work together in the complete pipeline. The notebook includes visual examples,
detailed explanations, and covers scenarios like low-quality images, multiple animal mentions,
empty text, and ambiguous classifications.

To run the demo:

```bash
jupyter notebook demo.ipynb
```

The notebook is structured in three main sections: image classification testing, NER testing, and
full pipeline integration. Each section includes both standard use cases and edge cases to
demonstrate the robustness of the solution.

## Component Documentation

Detailed documentation for each component is available in their respective directories. These
README files contain in-depth information about model architectures, training procedures, dataset
creation, performance metrics, and usage examples.

**Image Classification Component:** See `classifier/README.md` for complete documentation on the
VGG16-based classification model, including dataset collection process, architecture details,
training results (94.97% validation accuracy), and usage instructions.

**NER Component:** See `ner/README.md` for detailed information about the spaCy-based NER model,
including dataset creation methodology, training configuration, performance metrics (92.86%
precision, 97.50% recall, 95.12% F1-score), and usage examples.

## Requirements

The project dependencies are managed through `requirements.txt` and include the following main
packages:

- **tensorflow** - For image classification model training and inference
- **spacy** - For NER model training and text processing
- **pandas** - For data manipulation during training
- **scikit-learn** - For dataset splitting and evaluation metrics
- **Pillow** - For image loading and preprocessing
- **numpy** - For numerical operations
- **jupyter** - For running the demo notebook

Additional optional dependencies for data collection include Playwright for web scraping
functionality.

## Model Performance

The system achieves strong performance on both components. The image classifier demonstrates 94.97%
validation accuracy on a diverse set of animal images collected from multiple sources. The NER
model shows excellent entity extraction capabilities with 92.86% precision and 97.50% recall,
resulting in a 95.12% F1-score on the test set.

When combined in the pipeline, the system reliably verifies statements about animals in images,
handling various edge cases including multiple animal mentions, different capitalizations, and
ambiguous or low-confidence predictions.

## Training

Both models can be retrained on new data if needed. Training scripts are provided in their
respective component directories with configurable hyperparameters.

**Training the Image Classifier:**
```bash
python classifier/train_classifier.py \
    --dataset_path path/to/dataset \
    --epochs 100 \
    --batch_size 24 \
    --learning_rate 1e-4
```

**Training the NER Model:**
```bash
python ner/train_ner.py \
    --train_data data_for_ner/train.json \
    --test_data data_for_ner/test.json \
    --epochs 5 \
    --dropout 0.3
```

Detailed training instructions and parameter descriptions are available in the component-specific
README files.


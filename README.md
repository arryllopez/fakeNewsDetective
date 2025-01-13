# Fake News Detective

## Overview
Fake News Detective is a Python-based machine learning project aimed at detecting fake news using Natural Language Processing (NLP) techniques and pre-trained transformer models like BERT. The project is designed to help identify and classify news articles as "fake" or "true."

## Features
- **Dataset Support**: Compatible with the LIAR dataset and similar labeled datasets for fake news detection.
- **Transformer Models**: Uses Hugging Face's `transformers` library for pre-trained models like BERT.
- **Binary Classification**: Labels news articles as either "fake" or "true."
- **Early Stopping**: Implements early stopping to prevent overfitting during training.
- **Validation and Testing**: Includes functionality to evaluate the model's performance on validation and test datasets.
- **Customizable**: Easy to extend or fine-tune with other transformer models or datasets.

## Requirements
To run this project, make sure you have the following installed:

- Python 3.8 or later
- Required Python libraries (can be installed via `pip`):
  - `pandas`
  - `numpy`
  - `torch`
  - `transformers`
  - `scikit-learn`
  - `tqdm`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/arryllopez/fakeNewsDetective.git
   cd fakeNewsDetective
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. Download the LIAR dataset or any other dataset you want to use.
2. Place the dataset files in the `datasets` folder. Ensure the dataset files are in `.tsv` format with appropriate column names.

## Usage

### Running the Program
1. Open the main script (`detectingFakeNews.py`).
2. Modify paths to your dataset if needed.
3. Run the script:
   ```bash
   python detectingFakeNews.py
   ```

### Outputs
- The model saves the best weights during training to `best_model.pt`.
- After training, the final model and tokenizer are saved in the `fake_news_model` directory.
- Validation and test accuracies are printed during training and evaluation.

## File Structure
```
fakeNewsDetective/
|└── detectingFakeNews.py       # Main script for training and evaluation
|└── datasets/                 # Folder for storing datasets
|└── fake_news_model/         # Directory for saving the trained model
|└── requirements.txt         # Dependencies for the project
```

## Key Components
### Tokenization and Data Preparation
- Tokenizes text data using Hugging Face's `AutoTokenizer`.
- Converts text and labels into tensors for PyTorch.

### Model Training
- Utilizes a pre-trained BERT model (`bert-base-uncased`) for sequence classification.
- Implements a training loop with:
  - Cross-entropy loss.
  - AdamW optimizer.
  - Early stopping based on validation accuracy.

### Evaluation
- Computes validation and test accuracies using `scikit-learn`'s `accuracy_score`.

## How to Contribute
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them:
   ```bash
   git add .
   git commit -m "Add feature"
   git push origin feature-name
   ```
4. Create a pull request.

## Future Improvements
- Add support for multi-class classification.
- Experiment with other transformer models like RoBERTa or DistilBERT.
- Implement real-time fake news detection for live articles or social media data.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Hugging Face for their amazing `transformers` library.
- The authors of the LIAR dataset for providing a valuable resource for fake news detection research.


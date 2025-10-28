# ml-spam-detector

## Description
A Python Machine Learning project for spam email detection using scikit-learn. This project implements a spam classifier using Natural Language Processing (NLP) and Naive Bayes classification algorithm.

## Features
- Email/text spam detection
- Feature extraction using TfidfVectorizer
- Naive Bayes classification model
- Easy-to-use prediction interface

## Requirements
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/shithel9360/ml-spam-detector.git
cd ml-spam-detector
```

2. Install required packages:
```bash
pip install scikit-learn pandas numpy
```

## Usage

Run the spam detector:
```bash
python spam_detector.py
```

The script will:
1. Load sample training data
2. Extract features using TF-IDF vectorization
3. Train a Naive Bayes classifier
4. Make predictions on sample messages

## Example
```python
from spam_detector import SpamDetector

# Initialize the detector
detector = SpamDetector()

# Train the model
detector.train(messages, labels)

# Predict
result = detector.predict("Congratulations! You've won a free iPhone!")
print(result)  # Output: spam
```

## Project Structure
```
ml-spam-detector/
├── README.md
├── spam_detector.py    # Main spam detection implementation
└── requirements.txt    # Python dependencies
```

## How It Works
1. **Data Loading**: Loads training messages and their labels (spam/ham)
2. **Feature Extraction**: Converts text to numerical features using TF-IDF
3. **Model Training**: Trains a Multinomial Naive Bayes classifier
4. **Prediction**: Classifies new messages as spam or ham

## Future Enhancements
- Add support for loading datasets from CSV files
- Implement cross-validation for better accuracy metrics
- Add web interface for real-time spam detection
- Include more sophisticated NLP preprocessing

## License
MIT License

## Author
Developed as a machine learning demonstration project

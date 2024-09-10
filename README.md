# Emotion Recognition Model Using CNN

This project involves developing a Convolutional Neural Network (CNN) for recognizing human emotions from facial expressions using the FER2013 dataset.

## Overview

The model classifies images into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. It leverages advanced data preprocessing techniques and regularization strategies to optimize performance and generalization.

## Features

- **Emotion Classification**: Classifies images into seven emotion categories.
- **Data Preprocessing**: Includes normalization and augmentation for enhanced model robustness.
- **Regularization**: Uses Batch Normalization and Dropout to enhance model performance.
- **Model Training**: Built and trained using TensorFlow and Keras.

## Dataset

The FER2013 dataset is required for training and evaluation. It is not included in this repository due to its large size.

You can download the dataset from [Kaggle’s FER2013 page](https://www.kaggle.com/datasets/msambare/fer2013).

After downloading, place the dataset in the `dataset/FER2013` directory with the following structure:

```
dataset/
└── FER2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
```


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/emotion-recognition-model.git
   ```

2.	Navigate to the project directory:
    ```bash
    cd emotion-recognition-model
    ```

3.	Set up a virtual environment and install the required packages:
    ```bash
    python -m venv facenv
    source facenv/bin/activate  # On Windows use `facenv\Scripts\activate`
    pip install -r requirements.txt
    ```


## Usage

1. **Ensure the FER2013 dataset** is in the `dataset/FER2013` directory.

2. **Run emotion detection:**
   ```bash
   python recognition.py 
   ```

3. **Explore the notebook:**
    ```bash
    jupyter notebook recognition.ipynb
    ```

4.	**Model file:** emotion_detection.keras should be in the project directory for the scripts to function.



## Results

The model effectively classifies emotions from facial expressions and demonstrates practical insights into emotion recognition. The achieved performance is suitable for applications requiring emotion understanding.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
```
•	FER2013 dataset by Kaggle.
•	TensorFlow and Keras for their powerful libraries.
```
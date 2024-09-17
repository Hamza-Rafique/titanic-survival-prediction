# Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using machine learning models trained on the Titanic dataset.

## Project Structure

- `data/`: Contains the dataset files (train.csv, test.csv).
- `notebooks/`: Jupyter notebooks for analysis and model training.
- `models/`: Saved machine learning models.
- `scripts/`: Python scripts for preprocessing, training, and prediction.
- `submission/`: Submission files for Kaggle or similar platforms.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## How to Run

1. Install the required dependencies:

```python
  pip install -r requirements.txt
```

2. Run the preprocessing script:

```python
python scripts/preprocess.py
```

3. Train the model:

```python
python scripts/train.py
```

4. Make predictions:

```python
python scripts/predict.py
```


## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


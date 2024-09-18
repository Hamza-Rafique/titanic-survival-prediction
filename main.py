import os

# Step 1: Preprocess the data
os.system("python scripts/preprocess.py")

# # Step 2: Train the model
os.system("python scripts/train.py")

# # Step 3: Predict using the trained model
os.system("python scripts/predict.py")

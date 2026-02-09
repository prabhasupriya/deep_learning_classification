## Build a Deep Neural Network with TensorFlow/Keras for Binary Classification
### Project Objective 
The objective of this project is to develop and train a deep learning model to solve a binary classification problem using the TensorFlow and Keras frameworks. The project covers the end-to-end deep learning workflow: Exploratory Data Analysis (EDA), data preprocessing, neural network architecture design, and robust evaluation using validation strategies like early stopping and model checkpointing.
## Project Structure
deep_learning_classification/
├── data/
│   └── breast_cancer_data.csv    # The generated dataset file
├── models/
│   └── best_model.h5             # Saved weights of the best performing model
├── notebooks/
│   └── main_project.ipynb        # Fully documented Jupyter Notebook
├── README.md                     # Project documentation
└── requirements.txt              # List of dependencies
### Dataset & Preprocessing
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset.

EDA: Performed comprehensive analysis including class distribution visualization and feature correlation analysis.

Preprocessing Pipeline: * Handled potential missing values.

Scaled numerical features using StandardScaler to ensure optimal gradient descent.

Data Splitting: The dataset was split into Training (60%), Validation (20%), and Test (20%) sets using stratification to maintain class proportions.

### Neural Network Architecture
The model is a Sequential Neural Network designed as follows:

Input Layer: Dynamically shaped to match the 30 features of the dataset.

Hidden Layers: Three hidden layers (32, 16, and 8 neurons respectively) using the ReLU activation function.

Output Layer: A single neuron with a Sigmoid activation function for binary probability output.

Optimizer: Adam

Loss Function: Binary Crossentropy

### Model Training & Callbacks
To prevent overfitting and ensure the model generalizes well, the following callbacks were implemented:

EarlyStopping: Configured to monitor val_loss and stop training when performance plateaus (Patience = 10).

ModelCheckpoint: Configured to save only the best model weights based on the lowest validation loss.

Result: The model successfully converged and stopped at Epoch 59 to prevent overfitting.

### Evaluation and Results
The final model was evaluated on the held-out test set to ensure an unbiased estimate of performance:

Accuracy: 96%

ROC-AUC Score: 0.9931

F1-Score: 0.95 (Average)

Metrics Summary:
### Technical Stack
Language: Python 3.12

Frameworks: TensorFlow, Keras

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### How to Run
Ensure Python 3.12+ is installed.

Install dependencies:

Open notebooks/main_project.ipynb in VS Code or Jupyter to view the full execution and results.
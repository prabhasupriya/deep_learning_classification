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
## Dataset & Preprocessing
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset.

EDA: Comprehensive analysis including class distribution countplots and a feature correlation heatmap.

Preprocessing Pipeline:

Missing value verification and handling.

Scaling: Applied StandardScaler to normalize the 30 numerical features for optimal gradient descent.

Data Splitting: Dataset split into Training (60%), Validation (20%), and Test (20%) using stratification to maintain class balance.

## Neural Network Architecture
The model utilizes a Sequential architecture:

Input Layer: 30 neurons (matching dataset features).

Hidden Layers: 3 Dense layers with 32, 16, and 8 neurons respectively.

Activation: 'ReLU' for all hidden layers to prevent vanishing gradients.

Output Layer: 1 neuron with 'Sigmoid' activation for binary probability.

Optimization: Adam optimizer with Binary Crossentropy loss.

## Model Training & Callbacks
To ensure the model is "well-documented and rigorously evaluated," we implemented:

EarlyStopping: Monitored val_loss with patience=10. The model successfully converged and halted at Epoch 59.

ModelCheckpoint: Automatically saved the best weights to best_model.h5 based on validation performance.

## Evaluation and Results
Final evaluation was performed on the held-out Test Set (unseen data).

### Final Metrics:
## Technical Stack
Language: Python 3.12

Frameworks: TensorFlow, Keras

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Run
Ensure Python 3.12+ is installed.

Install dependencies:

Open notebooks/main_project.ipynb in VS Code or Jupyter to view the full execution, plots, and results.
##  Project Video Demonstration
Check out the full walkthrough of this project, including code execution and result analysis:
[Watch the Video on YouTube](https://youtu.be/c72WmdWD-vI)


##  Interactive Google Colab Notebook
You can run this project directly in your browser without any setup using the link below:
[Open in Google Colab](https://colab.research.google.com/drive/1MZWCL1F8bLIaqVzGfcHcJ9_rKbCwUzOV?usp=sharing)


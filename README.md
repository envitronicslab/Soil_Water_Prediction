## Predicting Soil Water Content and Potential Using Machine Learning

**Introduction:**

This repository showcases a comprehensive exploration of advanced machine learning algorithms applied to predict soil water content and potential using microclimate data. The dataset is a representative sample of microclimate data collected from six orchards in Washington State. Key variables include wind speed, solar radiation, relative humidity, air temperature, canopy temperature, and soil temperature.

By employing cutting-edge machine learning techniques, this project aims to develop accurate and reliable models capable of estimating soil water content and potential. The code encompasses data preprocessing, model training and evaluation, and result visualization to facilitate analysis and comparison of different algorithms.

**Development Environment:**

* **IDE:** Visual Studio Code
* **Programming Language:** Python
* **Environment Management:** Conda (Anaconda)

**Installation:**

1. Install Anaconda or Miniconda.
2. Create a new conda environment: `conda create -n your_env_name python=3.11`
3. Activate the environment: `conda activate your_env_name`
4. Install required libraries: `conda install pandas numpy scikit-learn tensorflow matplotlib tabulate joblib`

**Usage:**

1. Clone the repository.
2. Open the Jupyter Notebook file.
3. Run the cells to execute the code.

**Note:** Ensure you have the necessary data file ("royal_city_xx.csv") in the same directory as the Jupyter Notebook.

By following these steps, you should be able to successfully set up and run the project.

## Understanding the Task and Data

**Task:** Develop machine learning and neural network models to predict soil water content and potential based on given environmental variables.

**Data:** A CSV file named `royal_city_xx.csv` containing 9 columns: `julian_day`, `wind_speed`, `solar_radiation`, `relative_humidity`, `air_temp`, `canopy_temp_mean`, `soil_temp`, `water_content_mean`, `water_potential_mean`.

**Model Overview:**

This project employed a range of machine learning algorithms to predict soil water content and potential. The models included:

* **Linear Regression:** A classic statistical model that establishes a linear relationship between the features and the target variable.
* **Decision Tree Regressor:** A tree-based model that makes decisions based on a series of if-else questions.
* **Random Forest Regressor:** An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
* **Gradient Boosting Regressor:** Another ensemble method that builds a model iteratively, focusing on correcting the errors of previous models.
* **Support Vector Regression (SVR):** A kernel-based method that maps data to a higher-dimensional space to find a linear relationship.
* **Multi-Layer Perceptron (MLP) Regressor (scikit-learn):** A neural network model with multiple layers, suitable for complex patterns.
* **Custom TensorFlow Neural Network:** A neural network model built using TensorFlow.

**Goal:** Experiment with various models, evaluate their performance using machine learning metrics, and visualize the results.

## Python Code Implementation: Machine Learning Models

**Data Loading and Preprocessing:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from the CSV file
data = pd.read_csv("royal_city_xx.csv") # Daily average or late morning average csv file

# Extract features and target variables
X = data[['wind_speed', 'solar_radiation', 'relative_humidity', 'air_temp', 'canopy_temp_mean', 'soil_temp']]
y_water_content = data['water_content_mean']
y_water_potential = data['water_potential_mean']

# Normalize the features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets for model evaluation
X_train_water_content, X_test_water_content, y_train_water_content, y_test_water_content = train_test_split(X_scaled, y_water_content, test_size=0.2, random_state=42)
X_train_water_potential, X_test_water_potential, y_train_water_potential, y_test_water_potential = train_test_split(X_scaled, y_water_potential, test_size=0.2, random_state=42)
```

**Explanation:**

- The code loads the data from the "royal_city_xx.csv" file using pandas.
- It extracts the relevant features (X) and target variables (y) for water content and water potential.
- The features are normalized using `StandardScaler` to ensure consistent scaling and improve model performance.
- The data is split into training and testing sets to evaluate model performance on unseen data.

**Model Training and Evaluation:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# List of models to try
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), SVR(), MLPRegressor()]

# Train and evaluate models for water content prediction
results_water_content = {}
for model in models:
    mse, r2, y_pred = train_and_evaluate(model, X_train_water_content, y_train_water_content, X_test_water_content, y_test_water_content)
    results_water_content[str(model)] = {'MSE': mse, 'R2': r2, 'predicted_values': y_pred}

# ... (similar code for water potential prediction)
```

**Explanation:**

- A function `train_and_evaluate` is defined to train a given model, make predictions, and calculate evaluation metrics (MSE and R2).
- A list of different machine learning models is created, including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, and MLPRegressor.
- Each model is trained and evaluated on both water content and water potential data, and the results are stored in dictionaries.

**Model Evaluation and Results:**

```python
from tabulate import tabulate

# ... (create results tables as shown in previous responses)

# Print results
print("Water Content Prediction Results:")
print(tabulate(results_table_water_content, headers='keys', tablefmt='grid'))

print("\nWater Potential Prediction Results:")
print(tabulate(results_table_water_potential, headers='keys', tablefmt='grid'))
```

**Explanation:**

- The results are organized into tables using the `tabulate` library for better readability.
- The tables display the model name, MSE, and R2 values for each prediction task.

**Visualization:**

```python
import matplotlib.pyplot as plt

# Visualize results for water content prediction
fig, axs = plt.subplots(nrows=len(results_water_content), ncols=1, figsize=(10, 12))

for i, (model, results) in enumerate(results_water_content.items()):
    y_pred = results['predicted_values']
    axs[i].scatter(y_test_water_content, y_pred, label=model)
    axs[i].set_xlabel("Actual Water Content")
    axs[i].set_ylabel("Predicted Water Content")
    axs[i].set_title(f"Predicted vs. Actual Water Content - {model}")
    axs[i].legend()

plt.tight_layout()
plt.show()

# ... (similar code for water potential visualization)
```

**Explanation:**

- The code creates subplots to visualize the predicted vs. actual values for each model.
- Scatter plots are used to visualize the relationship between predicted and actual values.
- Labels, titles, and legends are added for clarity.

## Python Code Implementation: TensorFlow Neural Network

**Explanation:**

```python
import tensorflow as tf
print(tf.__version__)

# ... (other imports)
```

- This code imports necessary libraries:
    - `tensorflow` for building and training neural networks
    - `pandas` for data manipulation
    - `numpy` for numerical computations
    - `sklearn.model_selection` for data splitting
    - `sklearn.preprocessing` for data normalization
    - `sklearn.metrics` for evaluation metrics (commented out for now)
    - `tensorflow.keras` for building neural network models with Keras
    - `joblib` for saving and loading models (not used in this snippet)
    - `os` for file system operations (not used in this snippet)

```python
# Load the data from CSV file
data = pd.read_csv("royal_city_xx.csv") # Daily average or late morning average csv file

# Split data into features and target variables
X = data[['wind_speed', 'solar_radiation', 'relative_humidity', 'air_temp', 'canopy_temp_mean', 'soil_temp']]
y_water_content = data['water_content_mean']
y_water_potential = data['water_potential_mean']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train_water_content, X_test_water_content, y_train_water_content, y_test_water_content = train_test_split(X_scaled, y_water_content, test_size=0.2, random_state=42)
X_train_water_potential, X_test_water_potential, y_train_water_potential, y_test_water_potential = train_test_split(X_scaled, y_water_potential, test_size=0.2, random_state=42)
```

- These sections are the same as the previous explanation, loading and preprocessing the data.

```python
# Create a neural network model for water content prediction
model_nn_water_content = Sequential([
  Dense(64, activation='relu', input_dim=X.shape[1]),  # First hidden layer with 64 neurons and ReLU activation
  Dense(32, activation='relu'),                         # Second hidden layer with 32 neurons and ReLU activation
  Dense(1, activation='linear')                          # Output layer with 1 neuron and linear activation
])

model_nn_water_content.summary()
model_nn_water_content.compile(optimizer='adam', loss='mean_squared_error')
model_nn_water_content.fit(X_train_water_content, y_train_water_content, epochs=100, batch_size=32, validation_split=0.2)
```

- **Model Creation:**
    - A sequential neural network is defined using `Sequential`.
    - The model has three layers:
        - First hidden layer with 64 neurons and ReLU activation for non-linearity.
        - Second hidden layer with 32 neurons and ReLU activation.
        - Output layer with 1 neuron for water content prediction and linear activation for raw output.
    - `model.summary()` prints a summary of the model architecture.
- **Model Compilation:**
    - The model is compiled with the Adam optimizer for efficient weight updates.
    - The loss function is set to mean squared error (MSE) for regression tasks.
- **Model Training:**
    - The model is trained on the training data (`X_train_water_content`, `y_train_water_content`) for 100 epochs (iterations over the entire dataset).
    - A batch size of 32 is used to process data in smaller chunks during training.
    - `validation_split=0.2` allocates 20% of the training data for validation during training to monitor performance on unseen data.

```python
# Evaluate the model (commented out for now)
# mse_nn_water_content, r2_nn_water_content = model_nn_water_content.evaluate(X_test_water_content, y_test_water_content)
# results_water_content['Neural Network'] = {'MSE': mse_nn_water_content, 'R2': r2_nn_water_content}

# Calculate Metrics
y_pred_water_content = model_nn_water_content.predict(X_test_water_content)
r2_
```

## Saving and Loading Models

### Machine Learning Models

**Creating the Models Folder:**

```python
model_folder = "models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
```

- **Explanation:** This code creates a folder named "models" if it doesn't already exist. This ensures that the saved models are organized in a dedicated directory.

**Loading Saved Models Dynamically:**

```python
for model_name in [os.path.join(model_folder, f"{model.__class__.__name__}_model_water_content.pkl") for model in models]:
    model = joblib.load(model_name)
    mse, r2, y_pred = train_and_evaluate(model, X_train_water_content, y_train_water_content, X_test_water_content, y_test_water_content)
    results_water_content[str(model)] = {'MSE': mse, 'R2': r2}
```

**Explanation:**

- **Iterating over Model Names:** The loop iterates over a list of filenames constructed using the model class names and the "models" folder. This dynamically generates filenames based on the specific models used.
- **Loading Models:** For each filename, `joblib.load(model_name)` loads the corresponding saved model from the "models" folder.
- **Evaluating Models:** The loaded model is then evaluated using the `train_and_evaluate` function, and the results (MSE, R2) are stored in the `results_water_content` dictionary.

**Key Points:**

- This code demonstrates how to dynamically load multiple models from a specified folder without hardcoding individual filenames.
- It leverages the flexibility of `joblib` to load models based on their filenames.
- The code assumes that the models were saved with filenames that include their class names (e.g., "LinearRegression_model_water_content.pkl").
- This approach can be useful when you have a large number of models or when the model names are not known beforehand.


### TensorFlow Neural Network Models
This section of the code combines creating the models folder for saving and then loading the previously saved models. Here's a breakdown of each section:

**Creating the Models Folder (if needed):**

```python
model_folder = "models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
```

- **`model_folder = "models"`:** This line defines a variable named `model_folder` and assigns the string "models" to it. This variable will hold the name of the directory where you want to save your models.
- **`if not os.path.exists(model_folder):`:** This line checks if the directory specified by `model_folder` already exists. The `os.path.exists()` function returns `True` if the directory exists and `False` otherwise.
- **`os.makedirs(model_folder)`:** This line creates the directory specified by `model_folder` using the `os.makedirs()` function from the `os` library. This function can create multiple levels of directories if necessary.

In essence, this code snippet ensures that a directory named "models" exists. If the directory already exists, it does nothing. If the directory doesn't exist, it creates the directory for you. This is a good practice to organize your saved models and avoid issues if the directory is missing.

**Saving the Models (HDF5 format):**

```python
model_nn_water_content.save(os.path.join(model_folder, "model_nn_water_content.h5"))
model_nn_water_potential.save(os.path.join(model_folder, "model_nn_water_potential.h5"))
```

- **`model_nn_water_content.save(...)`:** This line saves the model named `model_nn_water_content` (trained neural network for water content prediction) using the `save()` method.
- **`os.path.join(model_folder, "...")`:** This part constructs the complete file path where the model will be saved. 
    - `os.path.join(model_folder, "...")` uses the `os.path.join()` function from the `os` library to combine the directory path (`model_folder`) and the filename (`"model_nn_water_content.h5"`). This ensures a proper path regardless of your operating system.
    - The filename specifies "model_nn_water_content.h5", indicating that the model is saved in the HDF5 format, a common format for saving TensorFlow models.
- **`model_nn_water_potential.save(...)`:** This line follows the same principle as the previous one, saving the `model_nn_water_potential` (trained neural network for water potential prediction) to a file named "model_nn_water_potential.h5" within the "models" directory.

Overall, this section uses the `save()` method on the trained models and constructs file paths within a dedicated "models" directory to keep the saved models organized.

**Loading the Created Neural Network Models:**

```python
model_nn_water_content = tf.keras.models.load_model(os.path.join(model_folder, "model_nn_water_content.h5"))
model_nn_water_potential = tf.keras.models.load_model(os.path.join(model_folder, "model_nn_water_potential.h5"))
```

- **`tf.keras.models.load_model(...)`:** This line is used to load previously saved models.
- **`os.path.join(model_folder, "...")`:** This part, similar to the saving section, constructs the complete file path for loading the models using `os.path.join()`.
- **Model Assignment:** The loaded models are then assigned to the variables `model_nn_water_content` and `model_nn_water_potential`, allowing you to use them for further predictions or analysis.

In summary, this code snippet combines managing the model directory, saving the trained models, and then loading them back for future use.

## Analyzing Model Performance: Daily Average Microclimate Data

**Performance Evaluation:**

The models were evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics. MSE measures the average squared difference between predicted and actual values, while R2 indicates the proportion of variance explained by the model.

| Model | Water Content (MSE, R2) | Water Potential (MSE, R2) |
|---|---|---|
| LinearRegression | (0.000451439, 0.665754) | (20.4275, 0.422086) |
| DecisionTreeRegressor | (0.0003, 0.777879) | (8.82254, 0.750402) |
| RandomForestRegressor | (0.000134631, 0.900319) | (4.74591, 0.865734) |
| GradientBoostingRegressor | (9.88234e-05, 0.926831) | (5.72953, 0.837906) |
| SVR | (0.00184444, -0.365631) | (27.3679, 0.225736) |
| MLPRegressor (scikit-learn) | (0.00398168, -1.94804) | (418.699, -10.8454) |
| TensorFlow Neural Network | (0.0036, -1.6845) | (83.5935, -1.3649) |

**Key Takeaways:**

- **Water Content:** Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor consistently outperformed other models, demonstrating excellent accuracy.
- **Water Potential:** Decision Tree Regressor, Random Forest Regressor, and GradientBoostingRegressor remained effective, suggesting their suitability for predicting water potential.
- **Neural Networks:** Both the scikit-learn MLPRegressor and the custom TensorFlow neural network struggled with water potential prediction, highlighting the potential challenges of neural networks for complex tasks.
- **Model Selection:** The choice of model might depend on specific requirements and considerations, such as interpretability, computational efficiency, and the complexity of the task.
- **Further Exploration:** For water potential prediction, exploring different neural network architectures, hyperparameter tuning, and feature engineering could potentially improve performance.

**Further Analysis:**

- Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor consistently demonstrated strong performance for both water content and potential.
- Neural networks might require further tuning and experimentation to effectively predict water potential.

## Analyzing Model Performance: : Late Morning Average Microclimate Data

**Performance Evaluation:**

The models were evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics.

| Model | Water Content (MSE, R2) | Water Potential (MSE, R2) |
|---|---|---|
| LinearRegression | (16.0563, 0.571595) | (1.46815, 0.861705) |
| DecisionTreeRegressor | (32.1413, 0.142422) | (0.812778, 0.923439) |
| RandomForestRegressor | (14.2428, 0.619981) | (0.504743, 0.952455) |
| GradientBoostingRegressor | (19.4914, 0.479941) | (0.580724, 0.945298) |
| SVR | (23.1705, 0.381778) | (3.03759, 0.713869) |
| MLPRegressor (scikit-learn) | (315.546, -7.41923) | (34.0868, -2.21087) |
| TensorFlow Neural Network | (0.0036, -1.6845) | (83.5935, -1.3649) |

**Key Takeaways:**

- **Water Content:** Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor demonstrated good performance, while Decision Tree Regressor and SVR struggled.
- **Water Potential:** RandomForestRegressor consistently outperformed other models, suggesting its suitability for predicting water potential.
- **Neural Networks:** Both the scikit-learn MLPRegressor and the custom TensorFlow neural network struggled with both water content and water potential prediction.

**Further Analysis:**

- The underperformance of the neural network models might be attributed to factors like hyperparameter tuning, data quality, or model complexity.
- Experimentation with different architectures, hyperparameters, and feature engineering techniques could potentially improve the performance of the neural networks.
- Consider exploring other machine learning algorithms or combining multiple models (ensembling) to achieve better results.

## Analyzing Model Performance: Daily vs. Late Morning Microclimate Data

**Performance Evaluation:**

The models were evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics.

| Model | Water Content (MSE, R2) | Water Potential (MSE, R2) |
|---|---|---|
| **Daily Average Data** | | |
| LinearRegression | (0.000451439, 0.665754) | (20.4275, 0.422086) |
| DecisionTreeRegressor | (0.0003, 0.777879) | (8.82254, 0.750402) |
| RandomForestRegressor | (0.000134631, 0.900319) | (4.74591, 0.865734) |
| GradientBoostingRegressor | (9.88234e-05, 0.926831) | (5.72953, 0.837906) |
| SVR | (0.00184444, -0.365631) | (27.3679, 0.225736) |
| MLPRegressor (scikit-learn) | (0.00398168, -1.94804) | (418.699, -10.8454) |
| TensorFlow Neural Network | (0.0036, -1.6845) | (83.5935, -1.3649) |
| **Late Morning Average Data** | | |
| LinearRegression | (16.0563, 0.571595) | (1.46815, 0.861705) |
| DecisionTreeRegressor | (32.1413, 0.142422) | (0.812778, 0.923439) |
| RandomForestRegressor | (14.2428, 0.619981) | (0.504743, 0.952455) |
| GradientBoostingRegressor | (19.4914, 0.479941) | (0.580724, 0.945298) |
| SVR | (23.1705, 0.381778) | (3.03759, 0.713869) |
| MLPRegressor (scikit-learn) | (315.546, -7.41923) | (34.0868, -2.21087) |
| TensorFlow Neural Network | (0.0036, -1.6845) | (83.5935, -1.3649) |

**Key Observations:**

- **Water Content:**
  - While most ML models performed well for both daily and late morning average data, the DecisionTreeRegressor underperformed for late morning average data.
  - Random Forest Regressor and Gradient Boosting Regressor consistently outperformed other models for daily average data, indicating their superior ability to capture the underlying patterns in the data.
  - For late morning average data, LinearRegression and RandomForestRegressor performed particularly well.
- **Water Potential:**
  - The choice of data type (daily vs. late morning average) had a significant impact on water potential prediction.
  - For daily average data, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor exhibited excellent performance.
  - For late morning average data, the performance of all ML models improved significantly.
- **Neural Networks:** Both the scikit-learn MLPRegressor and the custom TensorFlow neural network struggled with water potential prediction, regardless of data type.

**Conclusions:**

- **Data Type:** The choice of data type (daily vs. late morning average) had a significant impact on both water potential and water content predictions.
- **Model Selection:** Random Forest Regressor consistently demonstrated strong performance for both water content and potential.
- **Neural Networks:** Neural networks might require further tuning and experimentation to effectively predict water potential.

**Recommendations:**

- For water content prediction, Random Forest Regressor or Gradient Boosting Regressor can be used effectively with daily average data.
- For water potential prediction, consider using Decision Tree Regressor, Random Forest Regressor, or Gradient Boosting Regressor with late morning average data.
- Experiment with different neural network architectures, hyperparameters, and feature engineering techniques to improve their performance for water potential prediction.
- Continue exploring other machine learning algorithms and combining models to further enhance accuracy and robustness.

**Additional Insights:**

- The relatively high R-squared values for most models, especially for water content prediction, indicate that the models are able to explain a significant portion of the variance in the data.
- The lower MSE values for the top-performing models suggest that they are making more accurate predictions.
- The negative R-squared values for some neural network models highlight their poor performance, potentially due to overfitting or underfitting.

By carefully considering these insights and recommendations, you can make informed decisions about model selection and deployment for your specific use case.

## Updates

**Improved Model Performance:**

* Implemented a dedicated module to evaluate the MLPRegressor model for water content and potential prediction. This enhances code organization and facilitates future improvements.
* Optimized the TensorFlow/Keras Neural Network for water content and potential prediction.

**Performance Analysis:**

* The MLPRegressor shows strong performance for both water content and water potential prediction. Notably, it achieves a higher R2 value for water potential, indicating a better fit to the data. However, the higher MSE suggests the model might make slightly larger errors compared to water content prediction.
* The TensorFlow neural network performs well for water content prediction, with results comparable to the MLPRegressor. However, for water potential prediction, the MLPRegressor currently outperforms the neural network in terms of both MSE and R2. 

**Future Work:**

* We will continue to investigate and optimize the TensorFlow/Keras Neural Network for water potential prediction to improve its performance and potentially surpass the MLPRegressor.
* Further exploration of hyperparameter tuning and potentially different network architectures for the neural network might be beneficial.

**Benefits:**

These improvements contribute to a more robust and accurate prediction system for water content and potential. 
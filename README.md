## Predicting Soil Water Content and Potential Using Machine Learning

**Introduction:**

This repository showcases a comprehensive exploration of advanced machine learning algorithms applied to predict soil water content and potential using microclimate data. The dataset is a representative sample of microclimate data collected from six apple orchards in the United States. Key variables include wind speed, solar radiation, relative humidity, air temperature, canopy temperature, and soil temperature.

By employing cutting-edge machine learning techniques, this project aims to develop accurate and reliable models capable of estimating soil water content and potential. The code encompasses data preprocessing, model training and evaluation, and result visualization to facilitate analysis and comparison of different algorithms.

**Development Environment:**

* **IDE:** Visual Studio Code
* **Programming Language:** Python
* **Environment Management:** Conda (Anaconda)

**Installation:**

1. Install Anaconda or Miniconda.
2. Create a new conda environment: `conda create -n your_env_name python=3.xx`
3. Activate the environment: `conda activate your_env_name`
4. Install required libraries: `conda install pandas numpy scikit-learn tensorflow matplotlib tabulate joblib ...`

**Usage:**

1. Clone the repository.
2. Open the Jupyter Notebook file.
3. Run the cells to execute the code.

**Note:** Ensure you have the necessary data file ("royal_city_xx.csv") in the `microclimate_data` directory.

By following these steps, you should be able to successfully set up and run the project.

## Understanding the Task and Data

**Task:** Develop machine learning and neural network models to predict soil water content and potential based on given environmental variables.

**Data:** A CSV file named `royal_city_xx.csv` containing 9 columns: `julian_day`, `wind_speed`, `solar_radiation`, `relative_humidity`, `air_temp`, `canopy_temp_mean`, `soil_temp`, `water_content_mean`, `water_potential_mean`. 
**Note:** One of the data files also includes an additional column: `time`.

**Model Overview:**

This project employed a range of machine learning algorithms to predict soil water content and potential. The models included, but are not limited to:

* **Linear Regression:** A classic statistical model that establishes a linear relationship between the features and the target variable.
* **Decision Tree Regressor:** A tree-based model that makes decisions based on a series of if-else questions.
* **Random Forest Regressor:** An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
* **Gradient Boosting Regressor:** Another ensemble method that builds a model iteratively, focusing on correcting the errors of previous models.
* **Support Vector Regression (SVR):** A kernel-based method that maps data to a higher-dimensional space to find a linear relationship.
* **Multi-Layer Perceptron (MLP) Regressor (scikit-learn):** A neural network model with multiple layers, suitable for complex patterns.
* **Custom TensorFlow/Keras Neural Network:** A neural network model built using TensorFlow/Keras.

**Goal:** Experiment with various models, evaluate their performance using machine learning metrics, and visualize the results.
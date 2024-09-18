## Predicting Soil Water Content and Potential Using Machine Learning

**Introduction:**

This repository showcases a comprehensive exploration of advanced machine learning algorithms applied to predict soil water content and potential using microclimate data. The dataset is a representative sample of microclimate data collected from six apple orchards in Washington State. Key variables include wind speed, solar radiation, relative humidity, air temperature, canopy temperature, and soil temperature.

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
* **Custom TensorFlow/Keras Neural Network:** A neural network model built using TensorFlow/Keras.

**Goal:** Experiment with various models, evaluate their performance using machine learning metrics, and visualize the results.

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
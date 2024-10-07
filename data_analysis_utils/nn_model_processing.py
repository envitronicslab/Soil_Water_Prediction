import warnings
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers
from keras.regularizers import l1, l2
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from data_analysis_utils.model_processing import ModelProcessor

__author__ = "Y. Osroosh, Ph.D. <yosroosh@gmail.com>"

class NNModelProcessor(ModelProcessor):
    """
    A class to train and evaluate neural network models.
    
    Args:
        random_state (int, optional): The random seed for reproducible results. Defaults to 42.
        disable_warnings (bool, optional): Whether to disable warnings. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """

    def __init__(self, random_state=42, disable_warnings=True, verbose=True):
        super().__init__(random_state, disable_warnings, verbose)

        self.random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(42)        

        if disable_warnings:
            warnings.filterwarnings('ignore')

        self.verbose = verbose

        if verbose:
            print(f"Tensorflow Version: {tf.__version__}")

    def get_mlp_regressor_params_water_content(self):
        """
        Returns the parameter grid for MLPRegressor (Water Content Model).
        """
        return {
            # 'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],  # Experiment with different hidden layer configurations
            'hidden_layer_sizes': [(50, 50, 50, 50)],  
            # Try different activation functions
            # Choices are: 'identity', 'logistic', 'tanh', 'relu'. 
            # If you use an 'sgd' solver', try 'tanh' activation first.
            # If you use an 'adam' solver', try 'relu' activation first.
            # 'activation': ['relu', 'tanh', 'identity', 'logistic'],
            'activation': ['relu'], 
            # Try different solvers
            # Choices are: 'adam', 'sgd', or 'lbfgs'.     
            # 'solver': ['adam', 'sgd', 'lbfgs'], 
            'solver': ['lbfgs'], 
            # 'alpha': [0.0001, 0.001, 0.01],  # Experiment with different regularization parameters
            'alpha': [0.01],
            # 'learning_rate': ['constant', 'adaptive', 'invscaling'],  # Try different learning rate strategies
            'learning_rate': ['invscaling'],
            # 'learning_rate_init' : [0.001, 0.01, 0.05, 0.10], # Experiment with different learning rates
            'learning_rate_init' : [0.001],
            'max_iter' : [1000], # Set the maximum number of training iterations
            'shuffle' : [True],
            'n_iter_no_change' : [50], 
            'nesterovs_momentum' : [False],
            'verbose' : [False]
        }

    def get_mlp_regressor_params_water_potential(self):
        """
        Returns the parameter grid for MLPRegressor (Water Potential Model).
        """
        return {
            # 'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],  # Experiment with different hidden layer configurations
            'hidden_layer_sizes': [(50, 50, 50, 50)],  
            # Try different activation functions
            # Choices are: 'identity', 'logistic', 'tanh', 'relu'. 
            # If you use an 'sgd' solver', try 'tanh' activation first.
            # If you use an 'adam' solver', try 'relu' activation first.
            # 'activation': ['relu', 'tanh', 'identity', 'logistic'],
            'activation': ['tanh'], 
            # Try different solvers
            # Choices are: 'adam', 'sgd', or 'lbfgs'.     
            # 'solver': ['adam', 'sgd', 'lbfgs'], 
            'solver': ['sgd'], 
            # 'alpha': [0.0001, 0.001, 0.01],  # Experiment with different regularization parameters
            'alpha': [0.01],
            # 'learning_rate': ['constant', 'adaptive', 'invscaling'],  # Try different learning rate strategies
            'learning_rate': ['constant'],
            # 'learning_rate': ['adaptive'],
            # 'learning_rate_init' : [0.001, 0.01, 0.05, 0.10], # Experiment with different learning rates
            'learning_rate_init' : [0.01],
            'max_iter' : [1000], # Set the maximum number of training iterations
            'shuffle' : [True],
            'n_iter_no_change' : [50], 
            'nesterovs_momentum' : [False],
            'verbose' : [False]
        }

    def train_and_evaluate_mlp_regressor(self, target, X_train, y_train, X_test, y_test, verbose=2):
        """
        Trains and evaluates a Multi-Layer Perceptron (MLP) regressor using GridSearchCV.

        Args:
            X_train (pd.DataFrame): Training data features.
            y_train (pd.Series): Training data target variable.
            X_test (pd.DataFrame): Testing data features.
            y_test (pd.Series): Testing data target variable.
            target (str): Target variable name ('water_content' or 'water_potential').
            verbose (int): Verbosity level for GridSearchCV. Default is 2.

        Returns:
            tuple: A tuple containing two elements:
            - trained_model (sklearn.neural_network._multilayer_perceptron.MLPRegressor): The best trained MLPRegressor model.
            - results (dict): Dictionary containing evaluation results (MSE, R2, predicted values) for the best model.
        """

        if target == 'water_content':
            param_grid = self.get_mlp_regressor_params_water_content()
        elif target == 'water_potential':
            param_grid = self.get_mlp_regressor_params_water_potential()
        else:
            raise ValueError("Invalid target. Must be 'water_content' or 'water_potential'.")

        mlp_regressor = MLPRegressor()
        mlp_cv = GridSearchCV(mlp_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=verbose)
        
        print("\nTraining...")
        mlp_cv.fit(X_train, y_train)

        print("\nTraining complete. Evaluating...")
        best_mlp = mlp_cv.best_estimator_ # Get the best model from GridSearchCV
        
        # Train and evaluate the best model on entire training and testing data
        trained_model, result = self.train_and_evaluate(best_mlp, X_train, y_train, X_test, y_test) 

        # Remove hyperparameter details from the model name for cleaner presentation
        best_mlp = str(best_mlp)
        # Check if parentheses are present in the model name
        if "(" in best_mlp and ")" in best_mlp:            
            start_index = best_mlp.find("(") # Find the index of the first opening bracket            
            best_mlp = best_mlp[:start_index] # Extract the substring up to the first bracket (excluding the bracket)    

        results = {}
        results[best_mlp] = result # Store evaluation results for the best model

        # Print summary of the best model's hyperparameters
        best_params = mlp_cv.best_params_
        self.mlp_regressor_summary(best_params)

        # Print the best score achieved by GridSearchCV
        best_score = mlp_cv.best_score_
        print("\nBest Score", best_score)

        return trained_model, results
    
    @staticmethod
    def mlp_regressor_summary(params):
        """
        Prints a summary of an MLPRegressor model.

        Args:
            params (dict): A dictionary containing the best model parameters.
        """

        print("Model Summary (Best Params):")
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
        # Print the DataFrame as a table
        print(df.to_markdown(index=False))

    def define_tensorflow_keras_model_water_content(self, input_dim: int, units: int, learning_rate: float, l2_reg: float = 0.01) -> tf.keras.Model:
        """
        Defines a TensorFlow-Keras neural network model for water content prediction.

        This model uses a sequential architecture with ReLU activation for hidden layers,
        batch normalization for regularization, and dropout (commented out) for further potential improvements.

        Args:
            input_dim (int): The number of input features.
            units (int): The number of neurons in each hidden layer.
            learning_rate (float): The learning rate for the optimizer (Adam).
            l2_reg (float, optional): The L2 regularization strength. Defaults to 0.01.

        Returns:
            tf.keras.Model: A compiled TensorFlow-Keras neural network model.
        """
        
        model = Sequential([
            Dense(units, activation='relu', input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
            Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(momentum=0.8),
            # Dropout(0.2),
            Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(momentum=0.8),
            # Dropout(0.2),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        return model

    def define_tensorflow_keras_model_water_potential(self, input_dim: int, units: int, learning_rate: float, l2_reg: float = 0.001) -> tf.keras.Model:
        """
        Defines a TensorFlow-Keras neural network model for water potential prediction.

        This model uses a sequential architecture with tanh activation for hidden layers,
        batch normalization for regularization, and dropout (commented out) for further potential improvements.

        Args:
            input_dim (int): The number of input features.
            units (int): The number of neurons in each hidden layer.
            learning_rate (float): The learning rate for the optimizer (SGD).
            l2_reg (float, optional): The L2 regularization strength. Defaults to 0.001.

        Returns:
            tf.keras.Model: A compiled TensorFlow-Keras neural network model.
        """

        model = Sequential([
            Dense(units, activation='tanh', input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
            Dense(units, activation='tanh', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(momentum=0.8),
            # Dropout(0.2),
            Dense(units, activation='tanh', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(momentum=0.8),
            # Dropout(0.2),
            Dense(units, activation='tanh', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(momentum=0.8),
            # Dropout(0.2),
            Dense(1, activation='linear')
        ])

        model.compile(loss="mse", optimizer=optimizers.SGD(learning_rate=learning_rate))

        return model
    
    def train_and_evaluate_tensorflow_keras_model(self, model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Trains and evaluates a TensorFlow-Keras neural network model.

        Args:
            model (tf.keras.Model): The compiled TensorFlow-Keras neural network model.
            X_train (np.ndarray): The training input data.
            y_train (np.ndarray): The training target data.
            X_test (np.ndarray): The testing input data.
            y_test (np.ndarray): The testing target data.
            epochs (int, optional): The number of epochs for training. Defaults to 100.
            batch_size (int, optional): The batch size for training. Defaults to 32.

        Returns:
            dict: A dictionary containing the evaluation metrics (MSE and R2).
        """

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        evaluation_results = self.evaluate_tensorflow_keras_model(model, X_test, y_test)

        return evaluation_results
    
    def save_tensorflow_keras_models(self, models, save_path="models"):
        """
        Saves trained TensorFlow Keras neural network models to HDF5 format.

        Args:            
            models (dict, optional): A dictionary containing the trained TensorFlow Keras models. Keys should be descriptive names (e.g., "water_content", "water_potential") and values should be the corresponding TensorFlow model objects.
            save_path (str, optional): The directory to save the models. Defaults to "models".
        """

        # Create the save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save each model with a descriptive filename
        for model_group, model in models.items():      
            print(f"\nSave Model Group '{model_group}':")      
            if model is not None:
                filepath = os.path.join(save_path, f"Keras_model_nn_{model_group}.h5")
                model.save(filepath)
                print(f"Keras model '{model.__class__.__name__}' saved to: {filepath}")

    def load_tensorflow_keras_models(self, models, save_path="models"):
        """
        Loads trained TensorFlow Keras neural network models from HDF5 format.

        Args:
            save_path (str, optional): The directory containing the saved models. Defaults to "models".
            models (list, optional): A list of model names corresponding to the saved TensorFlow Keras models.

        Returns:
            dict: A dictionary containing the loaded TensorFlow Keras models. Keys are the model names and values are the corresponding TensorFlow model objects.
        """

        loaded_models = {}
        for name, model_list in models.items():
            if model_list:
                print(f"\nLoad Model Group '{name}':")
                if isinstance(model_list, list):  # Check if model_list is a list
                    pass  # Do nothing
                else:
                    filepath = os.path.join(save_path, f"Keras_model_nn_{name}.h5")
                    if os.path.exists(filepath):
                        loaded_models[name] = tf.keras.models.load_model(filepath)
                        print(f"Keras model loaded from: {filepath}")
                    else:
                        print(f"Keras model not found in {save_path}.")

        return loaded_models
    
    def evaluate_tensorflow_keras_model(self, model, X_test, y_test):
        """
        Evaluates a trained TensorFlow Keras neural network model on the provided test data.

        Args:
            model (tf.keras.Model): The loaded TensorFlow Keras model.
            X_test (np.ndarray): The testing input data.
            y_test (np.ndarray): The testing target data.

        Returns:
            dict: A dictionary containing the evaluation metrics (MSE and R2).
        """

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {}
        model_name = model.__class__.__name__
        results[model_name] = {'MSE': mse, 'R2': r2, 'predicted_values': y_pred}

        return results

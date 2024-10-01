import warnings
import joblib
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score
# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

__author__ = "Y. Osroosh, Ph.D. <yosroosh@gmail.com>"

class ModelProcessor:
    """
    A class to train and evaluate machine learning models.
    
    Args:
        random_state (int, optional): The random seed for reproducible results. Defaults to 42.
        disable_warnings (bool, optional): Whether to disable warnings. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """

    def __init__(self, random_state=42, disable_warnings=True, verbose=True):

        self.random_state = random_state  # Store the random state for later use
        np.random.seed(random_state)  # Set the random seed for NumPy's random number generator

        self.models = {
            "water_content": [
                LinearRegression(),
                DecisionTreeRegressor(random_state=random_state),
                RandomForestRegressor(random_state=random_state),
                GradientBoostingRegressor(random_state=random_state),
                ExtraTreesRegressor(n_estimators=100, random_state=random_state),
                AdaBoostRegressor(n_estimators=100, random_state=random_state),
                BaggingRegressor(n_estimators=100, random_state=random_state),
                SVR(),
            ],
            "water_potential": [
                LinearRegression(),
                DecisionTreeRegressor(random_state=random_state),
                RandomForestRegressor(random_state=random_state),
                GradientBoostingRegressor(random_state=random_state),
                ExtraTreesRegressor(n_estimators=100, random_state=random_state),
                AdaBoostRegressor(n_estimators=100, random_state=random_state),
                BaggingRegressor(n_estimators=100, random_state=random_state),
                SVR(),
            ],
        }

        if disable_warnings:        
            warnings.filterwarnings('ignore')  # early-stop warnings

        self.verbose = verbose

        if verbose:
            print(f"Random State: {random_state}")

    def train_and_evaluate_models(self, models_group, X_train, y_train, X_test, y_test):
        """
        Trains and evaluates multiple machine learning models on the given data.

        Args:
            models_group (str): The name of the list containing the machine learning models to train.
            X_train (pd.DataFrame): The training features (input data for training).
            y_train (pd.Series): The training labels (ground truth values for training).
            X_test (pd.DataFrame): The test features (input data for evaluation).
            y_test (pd.Series): The test labels (ground truth values for comparison).            

        Returns:
            A tuple containing:
            - A dictionary containing the trained models.
            - A dictionary containing evaluation metrics and predicted values for each model.
        """

        models = self.models.get(models_group, [])
        if not models:
            raise ValueError("Selected models group does not exist.")

        results = {}
        trained_models = []
        for model in tqdm(models, desc="Training Models"):
            trained_model, evaluation_results = self.train_and_evaluate(model, X_train, y_train, X_test, y_test)
            trained_models.append(trained_model)

            model_name = self.get_model_name_str(model)
                
            results[model_name] = evaluation_results

        return trained_models, results

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test):
        """
        Trains the model on the training data and evaluates its performance on the test data.

        Args:
            model (object): The machine learning model object to train and evaluate.
            X_train (pd.DataFrame): The training features (input data for training).
            y_train (pd.Series): The training labels (ground truth values for training).
            X_test (pd.DataFrame): The test features (input data for evaluation).
            y_test (pd.Series): The test labels (ground truth values for comparison).

        Returns:
        A tuple containing:
            - The trained model object.
            - A dictionary containing the evaluation metrics (e.g., MSE, R-squared) 
              returned by the `evaluate_model` function.
        """

        # Train the model
        trained_model = self.train_model(model, X_train, y_train)

        # Get evaluation metrics
        results = self.evaluate_model(trained_model, X_test, y_test)  

        return trained_model, results
    
    
    def train_model(self, model, X_train, y_train):
        """
        Trains a machine learning model on the given data.

        Args:
            model (object): The machine learning model object to train.
            X_train (pd.DataFrame): The training features (input data for training).
            y_train (pd.Series): The training labels (ground truth values for training).

        Returns:
            A tuple containing:
                - The trained model object.
        """

        # Train the model
        model.fit(X_train, y_train)

        return model

    def print_evaluation_results(self, results, title):
        """
        Prints the evaluation results for a given set of models.

        Args:
            results (dict): A dictionary containing the evaluation metrics and predicted values for each model.
            title (str): The title to print before the results.
        """

        print(f"\n{title} Prediction Results:")
        for model_name, result in results.items():            
            print(f"{model_name}:\tMSE={result['MSE']:.4f},\tR2={result['R2']:.4f}")

    def create_results_table(self, results, title):
        """
        Creates a table of model names, MSE, and R2 values.

        Args:
            results (dict): A dictionary containing the evaluation metrics and predicted values for each model.
            title (str): The title to print before the table.
        """

        model_names = []
        mse_values = []
        r2_values = []

        for model, result in results.items():
            model_name = self.get_model_name_str(model)
            model_names.append(model_name)
            mse_values.append(result['MSE'])
            r2_values.append(result['R2'])

        results_table = [{'Model': model_name, 'MSE': mse_value, 'R2': r2_value}
                        for model_name, mse_value, r2_value in zip(model_names, mse_values, r2_values)]
        
        print(f"\n{title} Prediction Results:")
        print(tabulate(results_table, headers='keys', tablefmt='grid'))

    def visualize_results(self, y_test, results, title):
        """
        Visualizes the predicted vs. actual values for a set of models.

        Args:
            y_test (pd.Series): The actual values used for comparison.
            results (dict): A dictionary containing the evaluation metrics and predicted values for each model.
            title (str): The title for the visualization.
        """

        # # Check if y_test is a NumPy array or a list of numbers
        # if not isinstance(y_test, (np.ndarray, list)) or not all(isinstance(x, (int, float)) for x in y_test):
        #     raise ValueError("y_test must be a NumPy array or a list of numbers.")

        num_subplots = len(results)  # Get the number of subplots based on the number of models

        # Create subplots based on the number of models
        if num_subplots == 1:
            fig, ax = plt.subplots()
            axs = [ax]  # Create a list of a single Axes object

            for i, (model_name, result) in enumerate(results.items()):
                model_predictions = result['predicted_values']
                axs[i].scatter(y_test, model_predictions, label=model_name)
                axs[i].set_xlabel("Actual Value")
                axs[i].set_ylabel("Predicted Value")
                axs[i].set_title(f"{title} - {model_name}")
                axs[i].legend()
                # Add text annotations with R² and MSE values
                axs[i].text(0.15, 0.75, f"R²: {result['R2']:.4f}", ha='left', va='top', transform=axs[i].transAxes, fontsize=10)
                axs[i].text(0.15, 0.70, f"MSE: {result['MSE']:.4f}", ha='left', va='top', transform=axs[i].transAxes, fontsize=10)
                ax.set_box_aspect(1) 
                axs[i].legend()                
                axs[i].grid(True) # Add grid to this subplot

                # Add diagonal line for perfect prediction
                xlims = axs[i].get_xlim()
                ylims = axs[i].get_ylim()
                xmin, xmax = min(xlims), max(xlims)
                ymin, ymax = min(ylims), max(ylims)
                axs[i].set_xlim(xmin, xmax)
                axs[i].set_ylim(ymin, ymax)
                axs[i].plot(xlims, ylims, 'k--', alpha=0.75, zorder=0, label="Perfect Prediction", color='gray')
                axs[i].legend()

        else:
            num_rows, num_cols = self.get_grid_shape(num_subplots)
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=self.get_figure_size(num_subplots), sharex=True, sharey=True)

            for i, (model_name, results) in enumerate(results.items()):
                model_predictions = results['predicted_values']
                axs[i // 2, i % 2].scatter(y_test, model_predictions, label=model_name)
                axs[i // 2, i % 2].set_xlabel("Actual Value")
                axs[i // 2, i % 2].set_ylabel("Predicted Value")
                axs[i // 2, i % 2].set_title(f"{model_name}")                
                axs[i // 2, i % 2].legend()
                # Add text annotations with R² and MSE values
                axs[i // 2, i % 2].text(0.15, 0.75, f"R²: {results['R2']:.4f}", ha='left', va='top', transform=axs[i // 2, i % 2].transAxes, fontsize=10)
                axs[i // 2, i % 2].text(0.15, 0.70, f"MSE: {results['MSE']:.4f}", ha='left', va='top', transform=axs[i // 2, i % 2].transAxes, fontsize=10)
                axs[i // 2, i % 2].set_box_aspect(1)    
                axs[i // 2, i % 2].legend()
                axs[i // 2, i % 2].grid(True)  # Add grid to this subplot                      

            # Add diagonal line for perfect prediction
            for ax in axs.flat: # Iterate through flattened subplots
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                xmin, xmax = min(xlims), max(xlims)
                ymin, ymax = min(ylims), max(ylims)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.plot(xlims, ylims, 'k--', alpha=0.75, zorder=0, label="Perfect Prediction", color='gray')
                ax.legend()                

            # Add overall title if there are multiple subplots
            if num_subplots > 1:
                fig.suptitle(title, fontsize=12)
                # set the spacing between subplots
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1, 
                                    right=2.5, 
                                    top=2.5, 
                                    wspace=0.1, 
                                    hspace=0.1)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_grid_shape(num_subplots):
        """
        Calculates the number of rows and columns for a given number of subplots.

        Args:
            num_subplots (int): The desired number of subplots.

        Returns:
            tuple: A tuple containing the number of rows and columns.
        """
        
        if num_subplots <= 1:
            return 1, 1
        else:
            num_rows = (num_subplots - 1) // 2 + 1  # Calculate the number of rows
            num_cols = 2  # Always maintain two subplots per row
            return num_rows, num_cols

    @staticmethod
    def get_figure_size(num_subplots):
        """
        Calculates the figure size based on the number of subplots.

        Args:
            num_subplots (int): The number of subplots.

        Returns:
            tuple: A tuple representing the figure width and height in inches.
        """

        rows, cols = ModelProcessor.get_grid_shape(num_subplots)
        width = cols * 6  # Adjust width based on number of columns
        height = rows * 4  # Adjust height based on number of rows

        return width, height
    
    def save_models(self, models, save_path='models'):
        """
        Saves the trained models to the specified path.

        Args:
            models (dict): A dictionary containing the trained models.
            save_path (str, optional): The path to save the trained models (default: 'models').
        """

        # Check if the models dictionary is empty
        if not models:
            raise ValueError("The 'models' dictionary is empty. No models to save.")

        # Create the save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save each model individually with a descriptive filename
        for model_group, models_ in models.items():            
            print(f"\nSave Model Group '{model_group}':") 
            if isinstance(models_, list):  # Check if models_ is a list
                for model in models_:         
                    model_name = self.get_model_name_str(model)
                    filepath = os.path.join(save_path, f"{model_name}_{model_group}_model.pkl")
                    joblib.dump(model, filepath)                
                    print(f"\t'{model_name}' saved to: {filepath}")
            else:
                # There's only one model in this group
                model = models_  # Access the single model directly
                model_name = self.get_model_name_str(model)
                filepath = os.path.join(save_path, f"{model_name}_{model_group}_model.pkl")
                joblib.dump(model, filepath)
                print(f"\t'{model_name}' saved to: {filepath}")

    def load_models(self, models, save_path="models"):
        """
        Loads saved models from the specified folder, filtering by the given model names.

        Args:
            models (dict): A dictionary containing the model names and corresponding lists of models to load.
            save_path (str, optional): The path to the directory where the models are saved (default: 'models').            

        Returns:
            dict: A dictionary containing the loaded models.
        """

        loaded_models = {}
        models_list = []
        for name, model_list in models.items():
            if model_list:
                print(f"\nLoad Model Group '{name}':")
                if isinstance(model_list, list):  # Check if model_list is a list
                    for model in model_list:
                        model_name = self.get_model_name_str(model)
                        filepath = os.path.join(save_path, f"{model_name}_{name}_model.pkl")
                        try:
                            model = joblib.load(filepath)
                            models_list.append(model)
                            print(f"\t'{filepath}' loaded.")
                        except FileNotFoundError:
                            print(f"\t'{model_name}' not found in {save_path}.")
                else:
                    model_name = self.get_model_name_str(model_list)
                    filepath = os.path.join(save_path, f"{model_name}_{name}_model.pkl")
                    try:
                        model = joblib.load(filepath)
                        models_list.append(model)
                        print(f"\t'{filepath}' loaded.")
                    except FileNotFoundError:
                        print(f"\t'{model_name}' not found in {save_path}.")

            loaded_models[name] = models_list
            models_list = [] # Empty the list for reuse

        return loaded_models

    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluates multiple loaded models on the provided test data.

        Args:
            models (dict): A dictionary containing the loaded machine learning models.
            X_test (pd.DataFrame): The testing input data (features).
            y_test (pd.Series): The testing target data (true labels).

        Returns:
            dict: A dictionary containing the evaluation metrics (MSE and R-squared) for each model.
        """
        
        evaluation_results = {}
        for model in models:
            model_name = self.get_model_name_str(model)
            evaluation_results[model_name] = self.evaluate_model(model, X_test, y_test)             
            
        return evaluation_results
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates a machine learning model's performance on the test data.

        Args:
            model (object): The trained machine learning model.
            X_test (pd.DataFrame): The test features (input data for evaluation).
            y_test (pd.Series): The test labels (ground truth values for comparison).

        Returns:
            dict: A dictionary containing the evaluation metrics (MSE and R-squared) and predicted values.
        """
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Create a dictionary to store the evaluation results
        results = {'MSE': mse, 'R2': r2, 'predicted_values': y_pred}  

        return results
    
    @staticmethod
    def get_model_name_str(model):
        """
        Extracts a clean string representation of a machine learning model.

        Args:
            model (object): The machine learning model object.

        Returns:
            str: A string representing the model name without parentheses.
        """
        
        model_name = str(model)

        # Remove parentheses from the model name if present
        if "(" in model_name and ")" in model_name:
            # model_name = re.sub(r"\(.*\)", "", model_name) # Does not work if there are multiple brackets in the model name 
            # Use regular expression to find the first opening bracket and extract everything before it
            model_name = re.search(r"^[^()]*", model_name).group()

        return model_name

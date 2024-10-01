import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__author__ = "Y. Osroosh, Ph.D. <yosroosh@gmail.com>"

class DataProcessor:
    """
    A class to process data used in training and evaluating machine learning models.
    
    Args:
        data_path (str): The path to the CSV file containing the data.
        skiprows (int, optional): The number of rows to skip at the beginning of the file. Defaults to 1.
        random_state (int, optional): The random seed for reproducible results. Defaults to 42.
        disable_warnings (bool, optional): Whether to disable warnings. Defaults to True.
        verbose (bool, optional): Whether to print verbose output during execution. Defaults to True.
    """

    def __init__(self, data_path, skiprows=1, random_state=42, disable_warnings=True, verbose=True):  

        self.random_state = random_state
        np.random.seed(random_state)      

        header = pd.read_csv(data_path, nrows=1)
        column_names = header.columns.tolist()

        self.data = pd.read_csv(data_path, skiprows=skiprows, header=None, names=column_names)

        if disable_warnings:        
            warnings.filterwarnings('ignore')  # early-stop warnings

        self.verbose = verbose

        if verbose:
            print(f"Random State: {random_state}")

    def get_data(self):
        return self.data
    
    def perform_eda(self):
        """
        Performs Exploratory Data Analysis (EDA) on the data.

        Args:
            None
        """
        print(self.data.head())
        print(self.data.info())
        print(self.data.describe())

    def pair_plots(self):
        '''
        Creates pair plots to explore correlations between features

        Args:
            None
        '''
        # Data visualization
        sns.pairplot(self.data)  
        plt.show()

    # ... Add functions for feature engineering, data cleaning, etc.

    def split_and_normalize_data(self, feature_columns, target_columns, test_size=0.2):
        """
        Splits the data into training and testing sets, prepares features and target variables,
        and normalizes the features using StandardScaler.

        Args:
            feature_columns (list): List of column names to be used as features (independent variables).
            target_columns (list): List of column names to be used as target variables (dependent variables).
            test_size (float, optional): Proportion of data for the test set (default: 0.2).

        Returns:
            tuple: A tuple containing the training and testing sets for features and target variables:
                - X_train_scaled (pd.DataFrame): Scaled training features.
                - X_test_scaled (pd.DataFrame): Scaled testing features.
                - y_train (pd.Series): Training target variables.
                - y_test (pd.Series): Testing target variables.
        """

        # Separate features and target variables
        X = self.data[feature_columns]
        y = self.data[target_columns]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)

        # Normalize the features using StandardScaler
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)  

        return X_train_scaled, X_test_scaled, y_train, y_test

    def normalize_features(self, X_train, X_test):
        """
        Normalizes the features using StandardScaler to center and scale them.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.

        Returns:
            tuple: A tuple containing the scaled training and testing features.
        """

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
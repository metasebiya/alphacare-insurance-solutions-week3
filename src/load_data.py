"""
load_data.py - Data loading function for insurance history data

This module contains functions to load insurance history data

Author: [Metasebiya Bizuneh]
Created: June 14, 2025
"""

import os
import pandas as pd


class DataLoader:
    # def __init__(self, customer_review):
    #     self.customer_review = customer_review

    def load_data(self, file_path):
        """
        Load a CSV file into a pandas DataFrame for financial analysis

        Parameters:
            file_path (str): Path to the CSV file (e.g., 'data/raw/{app_name}.csv')

        Returns:
            pd.DataFrame: Loaded customer review data as a DataFrame

        Raises:
            FileNotFoundError: If the specified file does not exist
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_csv(file_path, sep='|')
        print(df.head())
        return df

    def load_cleaned_data(self, file_path):
        """
        Load a CSV file into a pandas DataFrame for financial analysis

        Parameters:
            file_path (str): Path to the CSV file (e.g., 'data/raw/{app_name}.csv')

        Returns:
            pd.DataFrame: Loaded customer review data as a DataFrame

        Raises:
            FileNotFoundError: If the specified file does not exist
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_csv(file_path)
        print(df.head())
        return df


if __name__ == "__main__":
    df = "../data/raw/MachineLearningRating_v3.txt"

    data = DataLoader()
    all_data = data.load_data(df)
    print(all_data)
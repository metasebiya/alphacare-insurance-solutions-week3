import pandas as pd
from load_data import DataLoader
import os


class DataCleaner:

    def __init__(self, output_dir: str = "../data/processed/"):
        """
        Initializes the DataCleaner class with an output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure directory exists

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            print("\n⚠️ The DataFrame is empty.")
            return df

        print("\n📊 Initial Data Description:")
        print(df.describe(include='all'))
        print("\n📋 Initial Columns:", df.columns.tolist())



        # Shape and basic info
        print(f"\n🧱 Shape: {df.shape}")
        print(f"📦 Total Elements: {df.size}")
        print(f"\n📂 Data Types:\n{df.dtypes}")
        print("\n🧾 Missing Values Per Column:")
        print(df.isnull().sum())
        print(f"\n🔍 Duplicate Rows: {df.duplicated().sum()}")

        # Drop rows and columns with all NaN values
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        print(f"\n🔍 Duplicate Rows After dropping: {df.duplicated().sum()}")
        
        return df


if __name__ == "__main__":
    file_path = "../data/raw/MachineLearningRating_v3.txt"

    loader = DataLoader()
    data_dict = loader.load_data(file_path)
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_data(data_dict)
    print(f"\n🧼 Cleaned Data Preview for {cleaned_df}:")
    print(cleaned_df.head())

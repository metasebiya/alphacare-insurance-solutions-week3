import pandas as pd
from load_data import DataLoader
import os


class DataCleaner:
    dtype_map = {
        'UnderwrittenCoverID': 'int64',
        'PolicyID': 'int64',
        'TransactionMonth': 'datetime64[ns]',
        'IsVATRegistered': 'bool',
        'Citizenship': 'category',
        'LegalType' : 'category',
        'Title': 'object',
        'Language' : 'object',
        'Bank' : 'object',
        'AccountType' : 'category',
        'MaritalStatus': 'category',
        'Gender': 'category',
        'Country': 'object',
        'Province': 'object',
        'PostalCode': 'int64',
        'MainCrestaZone': 'object',
        'SubCrestaZone': 'object',
        'ItemType': 'category',
        'mmcode': 'float64',
        'VehicleType': 'category',
        'RegistrationYear': 'int64',
        'make': 'object',
        'Model': 'object',
        'Cylinders': 'float64',
        'cubiccapacity': 'float64',
        'kilowatts': 'float64',
        'bodytype': 'category',
        'NumberOfDoors': 'float64',
        'VehicleIntroDate': 'datetime64[ns]',
        'CustomValueEstimate': 'float64',
        'AlarmImmobiliser': 'object',
        'TrackingDevice': 'object',
        'CapitalOutstanding': 'object',
        'NewVehicle' : 'object',
        'WrittenOff': 'object',
        'Rebuilt': 'object',
        'Converted': 'object',
        'CrossBorder': 'object',
        'NumberOfVehiclesInFleet': 'float64',
        'SumInsured' : 'float64',
        'TermFrequency': 'object',
        'CalculatedPremiumPerTerm':'float64',
        'ExcessSelected': 'object',
        'CoverCategory': 'object',
        'CoverType': 'category',
        'CoverGroup': 'object',
        'Section' : 'object',
        'Product': 'object',
        'StatutoryClass': 'object',
        'StatutoryRiskType': 'category',
        'TotalPremium':'float64',
        'TotalClaims': 'float64'
    }


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

        #change datatype
        for col, dtype in self.dtype_map.items():
            if col in df.columns:
                if 'datetime' in dtype:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
            else:
                print(f"⚠️ Column '{col}' not found in DataFrame — skipping.")

        file_name = "MachineLearningRatingV3_cleaned.csv"
        file_path = os.path.join(self.output_dir, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🔄 Replacing existing file: {file_name}")
            df.to_csv(file_path, index=False)
            print(f"✅ Cleaned data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Failed to save cleaned data: {e}")

        return df


if __name__ == "__main__":
    file_path = "../data/raw/MachineLearningRating_v3.txt"

    loader = DataLoader()
    data_dict = loader.load_data(file_path)
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_data(data_dict)
    print(f"\n🧼 Cleaned Data Preview for {cleaned_df}:")
    print(cleaned_df.head())

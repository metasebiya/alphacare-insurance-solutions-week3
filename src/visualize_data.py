import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from load_data import DataLoader

# Set style for visualizations
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:

    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()  # Use .copy() to avoid SettingWithCopyWarning later

    def univariate_analysis(self):
        # Plot histogram of TotalPremium and TotalClaims
        sns.histplot(self.data['TotalPremium'], kde=True).set(title='TotalPremium Distribution')
        plt.show()

        sns.histplot(self.data['TotalClaims'], kde=True).set(title='TotalClaims Distribution')
        plt.show()

        # Categorical: Gender, Province
        sns.countplot(data=self.data, x='Gender').set(title='Gender Distribution')
        plt.show()

        sns.countplot(data=self.data, x='Province', order=self.data['Province'].value_counts().index).set(title='Policy Count by Province')
        plt.xticks(rotation=45)
        plt.show()

        plt.close()

    def bivariate_multivariate_analysis(self):
        # Loss Ratio calculation
        self.data['LossRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        province_loss = self.data.groupby('Province')['LossRatio'].mean().sort_values()

        province_loss.plot(kind='bar', title='Average Loss Ratio by Province')
        plt.ylabel('Loss Ratio')
        plt.xticks(rotation=45)
        plt.show()

        # Correlation heatmap for numeric variables
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr = self.data[num_cols].corr()

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

        plt.close()

    def outlier_detection(self):
        # Boxplots
        sns.boxplot(x=self.data['TotalClaims'])
        plt.title("Outliers in TotalClaims")
        plt.show()

        sns.boxplot(x=self.data['CustomValueEstimate'])
        plt.title("Outliers in CustomValueEstimate")
        plt.show()

        plt.close()

    def three_visualizations(self):
        # 1. Claims by Gender & Vehicle Type
        sns.boxplot(data=df, x='Gender', y='TotalClaims', hue='VehicleType')
        plt.title("Claim Amounts by Gender and Vehicle Type")
        plt.show()

        # # 2. Loss Ratio by Make
        top_makes = df['make'].value_counts().head(10).index
        sns.barplot(data=df[df['make'].isin(top_makes)], x='make', y='LossRatio')
        plt.xticks(rotation=45)
        plt.title("Loss Ratio by Vehicle Make")
        plt.show()

        # 3. Claim Trend Over Time
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df.groupby(df['TransactionMonth'].dt.to_period('M'))['TotalClaims'].sum().plot()
        plt.title("Total Claims Over Time")
        plt.xlabel("Month")
        plt.ylabel("Claims")
        plt.show()

        plt.close()

    def calc_loss_ratio(self):
        # 1. Basic Loss Ratio for the whole portfolio
        overall_loss_ratio = self.data['TotalClaims'].sum() / self.data['TotalPremium'].sum()
        print(f"Overall Loss Ratio: {overall_loss_ratio:.2f}")
        # Group and compute
        province_loss = self.data.groupby('Province')[['TotalClaims', 'TotalPremium']].sum()
        province_loss['LossRatio'] = province_loss['TotalClaims'] / province_loss['TotalPremium']

        self.data['LossRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        # Display
        print(province_loss[['LossRatio']].sort_values(by='LossRatio', ascending=False))

        # Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=province_loss.index, y=province_loss['LossRatio'])
        plt.title('Loss Ratio by Province')
        plt.ylabel('Loss Ratio')
        plt.xticks(rotation=45)
        plt.show()
        plt.close()

        #plot lossratio by gender
        gender_loss = self.data.groupby('Gender')[['TotalClaims', 'TotalPremium']].sum()
        gender_loss['LossRatio'] = gender_loss['TotalClaims'] / gender_loss['TotalPremium']

        print(gender_loss[['LossRatio']])

        sns.barplot(x=gender_loss.index, y=gender_loss['LossRatio'])
        plt.title('Loss Ratio by Gender')
        plt.ylabel('Loss Ratio')
        plt.show()
        plt.close()


        #Loss ratio by vehicle type
        vehicle_loss = self.data.groupby('VehicleType')[['TotalClaims', 'TotalPremium']].sum()
        vehicle_loss['LossRatio'] = vehicle_loss['TotalClaims'] / vehicle_loss['TotalPremium']

        print(vehicle_loss[['LossRatio']].sort_values(by='LossRatio', ascending=False))

        sns.barplot(x=vehicle_loss.index, y=vehicle_loss['LossRatio'])
        plt.title('Loss Ratio by Vehicle Type')
        plt.ylabel('Loself.datass Ratio')
        plt.xticks(rotation=45)
        plt.show()
        plt.close()

        return province_loss

    def plot_correlations(self):
        # 1. Ensure proper datetime
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])

        # 2. Group by PostalCode and Month
        monthly_zip = (
            self.data.groupby([self.data['TransactionMonth'].dt.to_period('M'), 'PostalCode'])[['TotalPremium', 'TotalClaims']]
            .sum()
            .reset_index()
        )

        # Convert Period to datetime for plotting
        monthly_zip['TransactionMonth'] = monthly_zip['TransactionMonth'].dt.to_timestamp()

        # 3. Explore a few zip codes (top 5 by volume)
        top_zips = self.data['PostalCode'].value_counts().head(5).index
        filtered_zip = monthly_zip[monthly_zip['PostalCode'].isin(top_zips)]

        # 4. Plot premium vs claims over time per zip
        for zip_code in top_zips:
            data = filtered_zip[filtered_zip['PostalCode'] == zip_code]

            plt.figure(figsize=(10, 4))
            plt.plot(data['TransactionMonth'], data['TotalPremium'], label='TotalPremium', marker='o')
            plt.plot(data['TransactionMonth'], data['TotalClaims'], label='TotalClaims', marker='x')
            plt.title(f'Monthly Premium vs Claims for Zip: {zip_code}')
            plt.xlabel('Month')
            plt.ylabel('Amount (ZAR)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # 5. Correlation per ZipCode (across time)
        # Create pivot table: rows = month, columns = zip, values = TotalPremium or Claims
        pivot_premium = monthly_zip.pivot(index='TransactionMonth', columns='PostalCode', values='TotalPremium')
        pivot_claims = monthly_zip.pivot(index='TransactionMonth', columns='PostalCode', values='TotalClaims')

        # Compute correlation across all zip codes (over time)
        corr_matrix = pivot_premium.corrwith(pivot_claims)

        # Show correlations between premium and claims per zip code
        print(corr_matrix.sort_values(ascending=False).dropna())

        # Optional: Heatmap for visual comparison
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame({'Correlation': corr_matrix}).sort_values(by='Correlation', ascending=False),
                    annot=True,
                    cmap='coolwarm')
        plt.title("Correlation between Monthly Premium and Claims per ZipCode")
        plt.show()
        plt.close()



if __name__ == "__main__":
    file_path = "../data/raw/MachineLearningRating_v3.txt"
    loader = DataLoader()
    df = loader.load_data(file_path)
    data = Visualizer(df)
    # visualize_1 = data.univariate_analysis()
    # visualize_2 = data.bivariate_multivariate_analysis()
    # visualize_3 = data.outlier_detection()
    data.calc_loss_ratio()
    visualize_4 = data.three_visualizations()
    data.plot_correlations()
    print(visualize_1)
    print(visualize_2)
    print(visualize_3)
    print(visualize_4)


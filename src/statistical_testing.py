import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from load_data import DataLoader

class StatisticalTester:
    def __init__(self, filepath):
        self.filepath = filepath
        self.loader = DataLoader()
        self.df = self.loader.load_cleaned_data(self.filepath)
        self.results = []

        # KPI calculations
        self.df['HasClaim'] = self.df['TotalClaims'].fillna(0).gt(0).astype(int)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        self.df['ClaimSeverity'] = self.df['TotalClaims'] / self.df['HasClaim'].replace(0, np.nan)


    def chi_squared_test(self, group_var, target_var, subset=None):
        data = self.df if subset is None else self.df[self.df[group_var].isin(subset)]
        contingency = pd.crosstab(data[group_var], data[target_var])
        chi2, p, dof, expected = chi2_contingency(contingency)
        return chi2, p

    def t_test(self, group_var, target_var, group_a, group_b):
        a_data = self.df[self.df[group_var] == group_a][target_var]
        b_data = self.df[self.df[group_var] == group_b][target_var]
        t_stat, p_val = ttest_ind(a_data.dropna(), b_data.dropna(), equal_var=False)
        return t_stat, p_val

    def run_tests(self):
        # Test 1: Province vs Claim Frequency
        prov_a, prov_b = 'Western Cape', 'Gauteng'
        _, p = self.chi_squared_test('Province', 'HasClaim', [prov_a, prov_b])
        self._log_result("Province vs Claim Frequency", "Chi-squared", prov_a, prov_b, p)

        # Test 2: Zip Code vs Claim Frequency
        zip_a, zip_b = 1000, 2000
        _, p = self.chi_squared_test('PostalCode', 'HasClaim', [zip_a, zip_b])
        self._log_result("Zip Code vs Claim Frequency", "Chi-squared", zip_a, zip_b, p)

        # Test 3: Zip Code vs Margin (t-test)
        t, p = self.t_test('PostalCode', 'Margin', zip_a, zip_b)
        self._log_result("Zip Code vs Margin", "t-test", zip_a, zip_b, p)

        # Test 4: Gender vs Claim Frequency
        gen_a, gen_b = 'Male', 'Female'
        _, p = self.chi_squared_test('Gender', 'HasClaim', [gen_a, gen_b])
        self._log_result("Gender vs Claim Frequency", "Chi-squared", gen_a, gen_b, p)

        # Test 5: Province vs Claim Severity
        t, p = self.t_test('Province', 'ClaimSeverity', prov_a, prov_b)
        self._log_result("Province vs Claim Severity", "t-test", prov_a, prov_b, p)

    def _log_result(self, test_name, method, group_a, group_b, p):
        self.results.append({
            "Test": test_name,
            "Method": method,
            "Groups": f"{group_a} vs {group_b}",
            "p-value": round(p, 4),
            "Conclusion": "Reject H₀" if p < 0.05 else "Fail to Reject H₀"
        })

    def get_results(self):
        return pd.DataFrame(self.results)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    filepath = "../data/processed/MachineLearningRatingV3_cleaned.csv"
    tester = StatisticalTester(filepath)
    tester.run_tests()
    results_df = tester.get_results()
    print(results_df)

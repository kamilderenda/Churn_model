import great_expectations as ge
from typing import Tuple, List

def validate_churn_data(df) -> Tuple[bool, List[str]]:
    print("Starting data validation...")
    ge_df=ge.dataset.PandasDataset(df)
    ge_df.expect_column_exists('CreditScore')
    ge_df.expect_column_exists('Geography')
    ge_df.expect_column_exists('Gender')
    ge_df.expect_column_exists('Age')
    ge_df.expect_column_exists('Tenure')
    ge_df.expect_column_exists('Balance')
    ge_df.expect_column_exists('NumOfProducts')
    ge_df.expect_column_exists('HasCrCard')
    ge_df.expect_column_exists('IsActiveMember')
    ge_df.expect_column_exists('EstimatedSalary')

    ge_df.expect_column_values_to_be_between('CreditScore', min_value=0)
    ge_df.expect_column_values_to_be_in_set('Geography', ['France', 'Spain', 'Germany'])
    ge_df.expect_column_values_to_be_in_set('Gender', ['Female', 'Male'])
    ge_df.expect_column_values_to_be_between('Age', min_value=18)
    ge_df.expect_column_values_to_be_between('Tenure', min_value=0)
    ge_df.expect_column_values_to_be_between('Balance', min_value=0)
    ge_df.expect_column_values_to_be_between('NumOfProducts', min_value=0)
    ge_df.expect_column_values_to_be_in_set('HasCrCard', [0, 1])
    ge_df.expect_column_values_to_be_in_set('IsActiveMember', [0, 1])
    ge_df.expect_column_values_to_be_between('EstimatedSalary', min_value=0)

    results=ge_df.validate()
    failed_expectations=[result['expectation_config']['expectation_type'] for result in results['results'] if not result['success']]
    if failed_expectations:
        print(f"Data validation failed for: {failed_expectations}")
        return False, failed_expectations
    else:
        print("Data validation passed successfully.")
        return True, []

# Error Analysis

| experiment                     | feature_set   | model                  |      MAE |       bias |   n |
|:-------------------------------|:--------------|:-----------------------|---------:|-----------:|----:|
| E7_US_harmonized_overlap_audit | weather_soil  | catboost_gpu           | 0.366291 | -0.095303  | 441 |
| E5_US_overlap_internal         | weather_soil  | catboost_gpu           | 0.366291 | -0.095303  | 441 |
| E7_US_processed_overlap_audit  | weather_soil  | catboost_gpu           | 0.366291 | -0.095303  | 441 |
| E7_US_harmonized_overlap_audit | weather_soil  | xgboost                | 0.367305 | -0.103036  | 441 |
| E5_US_overlap_internal         | weather_soil  | xgboost                | 0.367305 | -0.103036  | 441 |
| E7_US_processed_overlap_audit  | weather_soil  | xgboost                | 0.367305 | -0.103036  | 441 |
| E5_US_overlap_internal         | weather_soil  | hist_gradient_boosting | 0.372902 | -0.0974038 | 441 |
| E7_US_harmonized_overlap_audit | weather_soil  | hist_gradient_boosting | 0.372902 | -0.0974038 | 441 |
| E7_US_processed_overlap_audit  | weather_soil  | hist_gradient_boosting | 0.372902 | -0.0974038 | 441 |
| E7_US_harmonized_overlap_audit | weather_only  | xgboost                | 0.373544 | -0.103047  | 441 |
| E5_US_overlap_internal         | weather_only  | xgboost                | 0.373544 | -0.103047  | 441 |
| E7_US_processed_overlap_audit  | weather_only  | xgboost                | 0.373544 | -0.103047  | 441 |
| E7_US_harmonized_overlap_audit | weather_soil  | lightgbm               | 0.374894 | -0.113272  | 441 |
| E7_US_processed_overlap_audit  | weather_soil  | lightgbm               | 0.374894 | -0.113272  | 441 |
| E5_US_overlap_internal         | weather_soil  | lightgbm               | 0.374894 | -0.113272  | 441 |
| E4_AUS_overlap_internal        | weather_soil  | catboost_gpu           | 0.375512 | -0.179646  | 302 |
| E5_US_overlap_internal         | weather_only  | catboost_gpu           | 0.376001 | -0.0872978 | 441 |
| E7_US_harmonized_overlap_audit | weather_only  | catboost_gpu           | 0.376001 | -0.0872978 | 441 |
| E7_US_processed_overlap_audit  | weather_only  | catboost_gpu           | 0.376001 | -0.0872978 | 441 |
| E4_AUS_overlap_internal        | weather_only  | catboost_gpu           | 0.37862  | -0.179243  | 302 |
| E2_US_Wheat_internal           | weather_soil  | linear_svr             | 0.380917 | -0.0626367 | 148 |
| E7_US_harmonized_overlap_audit | weather_soil  | extra_trees            | 0.381064 | -0.111147  | 441 |
| E5_US_overlap_internal         | weather_soil  | extra_trees            | 0.381064 | -0.111147  | 441 |
| E7_US_processed_overlap_audit  | weather_soil  | extra_trees            | 0.381064 | -0.111147  | 441 |
| E4_AUS_overlap_internal        | weather_soil  | xgboost                | 0.382522 | -0.190195  | 302 |
| E7_US_harmonized_overlap_audit | weather_only  | extra_trees            | 0.383227 | -0.114963  | 441 |
| E5_US_overlap_internal         | weather_only  | extra_trees            | 0.383227 | -0.114963  | 441 |
| E7_US_processed_overlap_audit  | weather_only  | extra_trees            | 0.383227 | -0.114963  | 441 |
| E7_US_processed_overlap_audit  | weather_only  | hist_gradient_boosting | 0.386263 | -0.114423  | 441 |
| E5_US_overlap_internal         | weather_only  | hist_gradient_boosting | 0.386263 | -0.114423  | 441 |

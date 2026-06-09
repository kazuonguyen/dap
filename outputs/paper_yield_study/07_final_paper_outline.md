# Final Paper Outline

## Title

Cross-country crop yield prediction using climate and soil features: Australia and United States comparison for wheat and overlapping field crops

## Introduction

Motivate climate-driven crop yield prediction and cross-country generalization.

## Data

Describe Australia winter crop data, US harmonized data, target `yield_t_ha`, fair 1989-2021 comparison window, and leakage controls.

## Methods

Describe weather-only and weather+soil feature sets, random/time/country transfer splits, baseline, Ridge, Random Forest, Gradient Boosting, and CatBoost GPU.

## Results

Best model per experiment and feature set:

| experiment                     | feature_set   | split                                                | model                  |      MAE |     RMSE |        R2 |
|:-------------------------------|:--------------|:-----------------------------------------------------|:-----------------------|---------:|---------:|----------:|
| E1_AUS_Wheat_internal          | weather_only  | random_80_20                                         | gradient_boosting      | 0.270063 | 0.324055 | 0.92513   |
| E1_AUS_Wheat_internal          | weather_soil  | random_80_20                                         | gradient_boosting      | 0.274879 | 0.326869 | 0.923824  |
| E2_US_Wheat_internal           | weather_only  | random_80_20                                         | lasso                  | 0.339231 | 0.411048 | 0.771073  |
| E2_US_Wheat_internal           | weather_soil  | random_80_20                                         | linear_svr             | 0.335283 | 0.408061 | 0.774388  |
| E3_Wheat_transfer              | weather_only  | leave_country_out_train_Australia_test_United States | xgboost                | 0.627526 | 0.805678 | 0.0644433 |
| E3_Wheat_transfer              | weather_soil  | leave_country_out_train_Australia_test_United States | xgboost                | 0.631933 | 0.805051 | 0.065899  |
| E4_AUS_overlap_internal        | weather_only  | random_80_20                                         | xgboost                | 0.236732 | 0.345382 | 0.877409  |
| E4_AUS_overlap_internal        | weather_soil  | random_80_20                                         | catboost_gpu           | 0.236056 | 0.332956 | 0.886071  |
| E5_US_overlap_internal         | weather_only  | random_80_20                                         | xgboost                | 0.317732 | 0.413681 | 0.803378  |
| E5_US_overlap_internal         | weather_soil  | random_80_20                                         | catboost_gpu           | 0.300032 | 0.393608 | 0.821997  |
| E6_Combined_overlap            | weather_only  | random_80_20                                         | hist_gradient_boosting | 0.26412  | 0.349017 | 0.866994  |
| E6_Combined_overlap            | weather_soil  | random_80_20                                         | catboost_gpu           | 0.264466 | 0.350476 | 0.865879  |
| E7_US_harmonized_overlap_audit | weather_only  | random_80_20                                         | xgboost                | 0.317732 | 0.413681 | 0.803378  |
| E7_US_harmonized_overlap_audit | weather_soil  | random_80_20                                         | catboost_gpu           | 0.300032 | 0.393608 | 0.821997  |
| E7_US_processed_overlap_audit  | weather_only  | random_80_20                                         | xgboost                | 0.317732 | 0.413681 | 0.803378  |
| E7_US_processed_overlap_audit  | weather_soil  | random_80_20                                         | catboost_gpu           | 0.300032 | 0.393608 | 0.821997  |

## Discussion

Discuss transfer gap, domain shift, whether soil improves over weather-only, and whether wheat transfers more stably than multi-crop overlap.

## Conclusion

Summarize internal accuracy versus cross-country transfer behavior and future work on finer spatial data and management variables.

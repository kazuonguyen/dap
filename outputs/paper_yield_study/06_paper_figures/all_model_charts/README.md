# All Model Chart Gallery

Generated: 2026-06-09 15:27:17

Charts generated: 278
Models covered: 24
Scenarios covered: 36

## Chart Counts

- feature_importance: 19
- global: 19
- heatmap: 24
- scenario_metric: 108
- scenario_prediction: 108

## Best 15 Runs by RMSE

| experiment              | dataset        | feature_set   | split        | model                  |    MAE |   RMSE |     R2 |
|:------------------------|:---------------|:--------------|:-------------|:-----------------------|-------:|-------:|-------:|
| E1_AUS_Wheat_internal   | aus_wheat      | weather_only  | random_80_20 | gradient_boosting      | 0.2701 | 0.3241 | 0.9251 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_soil  | random_80_20 | gradient_boosting      | 0.2749 | 0.3269 | 0.9238 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_only  | random_80_20 | ngboost                | 0.2793 | 0.3269 | 0.9238 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_soil  | random_80_20 | ngboost                | 0.2783 | 0.3276 | 0.9235 |
| E4_AUS_overlap_internal | aus_overlap    | weather_soil  | random_80_20 | catboost_gpu           | 0.2361 | 0.3330 | 0.8861 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_only  | random_80_20 | adaboost_tree          | 0.2646 | 0.3345 | 0.9202 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_soil  | random_80_20 | adaboost_tree          | 0.2612 | 0.3367 | 0.9192 |
| E4_AUS_overlap_internal | aus_overlap    | weather_soil  | random_80_20 | xgboost                | 0.2316 | 0.3371 | 0.8832 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_soil  | random_80_20 | extra_trees            | 0.2716 | 0.3388 | 0.9181 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_only  | random_80_20 | extra_trees            | 0.2779 | 0.3436 | 0.9158 |
| E4_AUS_overlap_internal | aus_overlap    | weather_only  | random_80_20 | xgboost                | 0.2367 | 0.3454 | 0.8774 |
| E1_AUS_Wheat_internal   | aus_wheat      | weather_only  | random_80_20 | bagging_tree           | 0.2921 | 0.3483 | 0.9135 |
| E6_Combined_overlap     | aus_us_overlap | weather_only  | random_80_20 | hist_gradient_boosting | 0.2641 | 0.3490 | 0.8670 |
| E6_Combined_overlap     | aus_us_overlap | weather_only  | random_80_20 | lightgbm               | 0.2613 | 0.3501 | 0.8662 |
| E6_Combined_overlap     | aus_us_overlap | weather_soil  | random_80_20 | catboost_gpu           | 0.2645 | 0.3505 | 0.8659 |

## Files

- Manifest CSV: `chart_manifest.csv`
- Global charts: `global/`
- Heatmaps: `heatmaps/`
- Scenario metric charts: `scenario_metrics/`
- Prediction and residual charts: `scenario_predictions/`
- Feature-importance charts: `feature_importance/`

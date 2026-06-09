# Feature Schema Report

## Leakage Policy

The target is `yield_t_ha`. Yield originals, production, area, harvested-area, USDA metadata, unit/value fields, and any target-derived columns are excluded.

## Feature Sets

### weather_only

- Numeric features: 72
- Categorical features: country, region, crop
- Columns: year_start, avg_radiation_may_oct, avg_tmax_may_oct, avg_tmean_may_oct, avg_tmin_may_oct, dry_days_aug, dry_days_jul, dry_days_jun, dry_days_may, dry_days_may_oct, dry_days_oct, dry_days_sep, heat_days_25_aug, heat_days_25_jul, heat_days_25_jun, heat_days_25_may, heat_days_25_may_oct, heat_days_25_oct, heat_days_25_sep, heat_days_30_aug, heat_days_30_jul, heat_days_30_jun, heat_days_30_may, heat_days_30_may_oct, heat_days_30_oct, heat_days_30_sep, heat_days_35_aug, heat_days_35_jul, heat_days_35_jun, heat_days_35_may, heat_days_35_may_oct, heat_days_35_oct, heat_days_35_sep, radiation_may_oct, radiation_mj_m2_sum_aug, radiation_mj_m2_sum_jul, radiation_mj_m2_sum_jun, radiation_mj_m2_sum_may, radiation_mj_m2_sum_oct, radiation_mj_m2_sum_sep, rain_may_oct, rain_mm_sum_aug, rain_mm_sum_jul, rain_mm_sum_jun, rain_mm_sum_may, rain_mm_sum_oct, rain_mm_sum_sep, tmax_c_mean_aug, tmax_c_mean_jul, tmax_c_mean_jun, tmax_c_mean_may, tmax_c_mean_oct, tmax_c_mean_sep, tmean_c_mean_aug, tmean_c_mean_jul, tmean_c_mean_jun, tmean_c_mean_may, tmean_c_mean_oct, tmean_c_mean_sep, tmin_c_mean_aug, tmin_c_mean_jul, tmin_c_mean_jun, tmin_c_mean_may, tmin_c_mean_oct, tmin_c_mean_sep, wet_days_aug, wet_days_jul, wet_days_jun, wet_days_may, wet_days_may_oct, wet_days_oct, wet_days_sep, country, region, crop

### weather_soil

- Numeric features: 120
- Categorical features: country, region, crop
- Columns: year_start, avg_radiation_may_oct, avg_tmax_may_oct, avg_tmean_may_oct, avg_tmin_may_oct, dry_days_aug, dry_days_jul, dry_days_jun, dry_days_may, dry_days_may_oct, dry_days_oct, dry_days_sep, heat_days_25_aug, heat_days_25_jul, heat_days_25_jun, heat_days_25_may, heat_days_25_may_oct, heat_days_25_oct, heat_days_25_sep, heat_days_30_aug, heat_days_30_jul, heat_days_30_jun, heat_days_30_may, heat_days_30_may_oct, heat_days_30_oct, heat_days_30_sep, heat_days_35_aug, heat_days_35_jul, heat_days_35_jun, heat_days_35_may, heat_days_35_may_oct, heat_days_35_oct, heat_days_35_sep, radiation_may_oct, radiation_mj_m2_sum_aug, radiation_mj_m2_sum_jul, radiation_mj_m2_sum_jun, radiation_mj_m2_sum_may, radiation_mj_m2_sum_oct, radiation_mj_m2_sum_sep, rain_may_oct, rain_mm_sum_aug, rain_mm_sum_jul, rain_mm_sum_jun, rain_mm_sum_may, rain_mm_sum_oct, rain_mm_sum_sep, tmax_c_mean_aug, tmax_c_mean_jul, tmax_c_mean_jun, tmax_c_mean_may, tmax_c_mean_oct, tmax_c_mean_sep, tmean_c_mean_aug, tmean_c_mean_jul, tmean_c_mean_jun, tmean_c_mean_may, tmean_c_mean_oct, tmean_c_mean_sep, tmin_c_mean_aug, tmin_c_mean_jul, tmin_c_mean_jun, tmin_c_mean_may, tmin_c_mean_oct, tmin_c_mean_sep, wet_days_aug, wet_days_jul, wet_days_jun, wet_days_may, wet_days_may_oct, wet_days_oct, wet_days_sep, soil_bulk_density_0_5, soil_bulk_density_100_200, soil_bulk_density_15_30, soil_bulk_density_30_60, soil_bulk_density_5_15, soil_bulk_density_60_100, soil_cec_0_5, soil_cec_100_200, soil_cec_15_30, soil_cec_30_60, soil_cec_5_15, soil_cec_60_100, soil_clay_0_5, soil_clay_100_200, soil_clay_15_30, soil_clay_30_60, soil_clay_5_15, soil_clay_60_100, soil_nitrogen_0_5, soil_nitrogen_100_200, soil_nitrogen_15_30, soil_nitrogen_30_60, soil_nitrogen_5_15, soil_nitrogen_60_100, soil_ph_0_5, soil_ph_100_200, soil_ph_15_30, soil_ph_30_60, soil_ph_5_15, soil_ph_60_100, soil_sand_0_5, soil_sand_100_200, soil_sand_15_30, soil_sand_30_60, soil_sand_5_15, soil_sand_60_100, soil_silt_0_5, soil_silt_100_200, soil_silt_15_30, soil_silt_30_60, soil_silt_5_15, soil_silt_60_100, soil_soc_0_5, soil_soc_100_200, soil_soc_15_30, soil_soc_30_60, soil_soc_5_15, soil_soc_60_100, country, region, crop

## Raw Leakage Candidates Dropped

- aus: 0 candidate columns: none
- us_harm_wheat: 0 candidate columns: none
- us_harm_overlap: 0 candidate columns: none
- us_processed_wheat: 15 candidate columns: CV_pct, Value, class_desc, commodity_desc, domain_desc, group_desc, prodn_practice_desc, sector_desc, short_desc, source_desc, statisticcat_desc, unit_desc, util_practice_desc, yield_original, yield_unit
- us_processed_all: 15 candidate columns: CV_pct, Value, class_desc, commodity_desc, domain_desc, group_desc, prodn_practice_desc, sector_desc, short_desc, source_desc, statisticcat_desc, unit_desc, util_practice_desc, yield_original, yield_unit

## Soil Mapping

AUS uppercase soil columns and US SoilGrids-style columns are mapped to common concept names for clay, sand, silt, SOC, nitrogen, pH, bulk density, and CEC.

## CatBoost GPU

- CatBoost requested: True
- CatBoost available in run: True
- GPU fallback allowed: True
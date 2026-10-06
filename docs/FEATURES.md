# Features

The dev build has 339 features in 14 groups. The windows and thresholds of each group are the constants at the top of its module, next to the formula of each feature.

A per-ETF feature becomes its cross-sectional rank among the tradable ETFs, minus 0.5. A global feature becomes its expanding past-only percentile, minus 0.5. In the pca_etf group, the names that start with `etf_` are global.


| Group | Count | Kind | Code |
|---|---|---|---|
| price | 35 | per-ETF | etf66/features/price.py (momentum) |
| volatility | 30 | per-ETF | etf66/features/price.py (volatility) |
| drawdown | 16 | per-ETF | etf66/features/price.py (drawdown) |
| trend | 27 | per-ETF | etf66/features/price.py (trend) |
| candles | 27 | per-ETF | etf66/features/bars.py (candles) |
| gaps | 14 | per-ETF | etf66/features/bars.py (gaps) |
| compression | 13 | per-ETF | etf66/features/bars.py (compression) |
| volume | 22 | per-ETF | etf66/features/bars.py (volume) |
| relative | 24 | per-ETF | etf66/features/cross.py (relative) |
| screeners | 17 | per-ETF | etf66/features/cross.py (screeners) |
| pca_etf | 48 | mixed (5 per-ETF, 43 global) | etf66/features/state.py (etf_factor_features, etf_state) |
| pca_stock | 32 | global | etf66/features/state.py (stock_state) |
| macro | 29 | global | etf66/features/macro.py (macro) |
| calendar | 5 | global | etf66/features/macro.py (calendar) |

## price

`ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_21`, `ret_63`, `ret_126`, `ret_252`, `mom_12_1`, `mom_6_1`, `mom_6_3`, `sharpe_21`, `sharpe_63`, `sharpe_126`, `sharpe_252`, `trend_skip21_63`, `trend_skip21_126`, `trend_skip21_252`, `accel_5_21`, `accel_21_63`, `accel_63_252`, `momentum_pullback`, `drawdown_adjusted_momentum`, `multi_horizon_positive_share`, `post_crash_bounce`, `slope_log_price_10`, `slope_log_price_21`, `slope_log_price_63`, `slope_accel_21`, `up_day_share_21`, `up_day_share_63`, `up_day_share_150`, `upside_capture_63`, `up_streak`, `down_streak`

## volatility

`vol_10`, `vol_21`, `vol_63`, `vol_126`, `downside_vol_21`, `downside_vol_63`, `up_down_vol_ratio_63`, `vol_ratio_21_63`, `vol_ratio_63_252`, `vol_change_21`, `vol_of_vol_63`, `parkinson_10`, `parkinson_21`, `parkinson_50`, `garman_klass_21`, `rogers_satchell_21`, `yang_zhang_21`, `yang_zhang_63`, `atr_14`, `atr_63`, `atr_ratio_14_63`, `vol_pct_self_21`, `vol_pct_self_63`, `skew_63`, `kurt_63`, `skew_126`, `kurt_126`, `worst_day_21`, `expected_shortfall_63`, `tail_balance_252`

## drawdown

`drawdown_22`, `drawdown_63`, `drawdown_252`, `distance_from_low_63`, `ulcer_14`, `ulcer_50`, `max_drawdown_63`, `max_drawdown_252`, `days_since_high_63`, `days_since_high_252`, `days_since_low_63`, `underwater_share_63`, `underwater_area_63`, `range_position_20`, `range_position_63`, `range_position_252`

## trend

`ma_gap_10`, `ma_gap_20`, `ma_gap_50`, `ma_gap_100`, `ma_gap_200`, `ema_ratio_5_20`, `ema_ratio_10_50`, `ema_ratio_20_100`, `ema_ratio_50_200`, `sma_slope_20`, `sma_slope_50`, `ppo_12_26`, `macd_hist`, `rsi_2`, `rsi_5`, `rsi_14`, `rsi_21`, `stochastic_14`, `cci_20`, `adx_14`, `aroon_osc_25`, `bollinger_pctb_20`, `bollinger_width_20`, `efficiency_21`, `efficiency_63`, `time_above_ma200_126`, `run_above_ma200`

## candles

`candle_body_1`, `candle_upper_wick_1`, `candle_lower_wick_1`, `candle_close_location_1`, `candle_return_1`, `candle_body_5`, `candle_upper_wick_5`, `candle_lower_wick_5`, `candle_close_location_5`, `candle_return_5`, `candle_body_21`, `candle_upper_wick_21`, `candle_lower_wick_21`, `candle_close_location_21`, `candle_return_21`, `candle_body_1_mean_5`, `candle_body_1_mean_21`, `candle_upper_wick_1_mean_5`, `candle_upper_wick_1_mean_21`, `candle_lower_wick_1_mean_5`, `candle_lower_wick_1_mean_21`, `candle_close_location_1_mean_5`, `candle_close_location_1_mean_21`, `range_1`, `range_mean_21`, `range_ratio_5_vs_prior_5`, `clv_imbalance_21`

## gaps

`gap_1`, `gap_mean_21`, `gap_abs_sum_21`, `gap_balance_63`, `gap_recovery_average_63`, `gap_down_recovery_rate_126`, `gap_follow_through_63`, `overnight_sum_21`, `intraday_sum_21`, `overnight_sum_63`, `intraday_sum_63`, `overnight_share_63`, `overnight_minus_intraday_63`, `gap_fill_rate_66`

## compression

`range_expansion_21`, `hl_compression_21_126`, `bollinger_squeeze_63`, `donchian_position_20`, `donchian_position_55`, `donchian_width_20`, `vcp_touch_count_21`, `nr7_count_21`, `inside_bar_count_21`, `breakout_up_20`, `breakdown_20`, `breakout_up_50`, `breakdown_50`

## volume

`dollar_volume_z_21`, `dollar_volume_z_63`, `volume_ratio_5_63`, `volume_ratio_21_252`, `volume_migration_21`, `volume_block_product`, `up_down_volume_ratio_21`, `up_down_volume_ratio_63`, `climactic_volume_21`, `price_volume_trend_21`, `obv_slope_21`, `obv_slope_63`, `chaikin_money_flow_20`, `money_flow_index_14`, `amihud_63`, `range_per_dollar_21`, `liquidity_relief`, `kyle_proxy_63`, `volume_attention_x1`, `volume_attention_x2`, `volume_attention_x4`, `log_adv`

## relative

`beta_spy_63`, `beta_spy_252`, `down_beta_spy_63`, `beta_decoupling_63_252`, `corr_spy_21`, `corr_spy_63`, `up_capture_63`, `down_capture_63`, `relative_return_spy_5`, `relative_return_spy_21`, `relative_return_spy_63`, `residual_momentum_63`, `block_relative_return_5`, `block_relative_return_21`, `block_relative_return_63`, `block_z_return_5`, `corr_tlt_63`, `corr_hyg_63`, `corr_gld_63`, `corr_uup_63`, `corr_vxx_63`, `beta_tlt_63`, `beta_uup_63`, `lead_corr_spy_lag1_63`

## screeners

`x3_vol_expansion_ratio`, `x3_flag`, `o2_cs_spread_63`, `o2_flag`, `o3_roll_spread_63`, `o3_flag`, `p2_flag`, `x5_tail_ratio_pct`, `x5_flag`, `r1_flag`, `g5_gap_down_count_63`, `g5_recovery_rate_126`, `g5_flag`, `a4_upside_vol_pct`, `a4_flag`, `m2_flag`, `lens_vote_count`

## pca_etf

`pc1_loading`, `pc2_loading`, `factor_share_top3`, `idiosyncratic_share`, `pc1_loading_change_20`, `etf_rotation_overlap_20`, `etf_idiosyncratic_share_median`, `etf_eff_n_63`, `etf_k50_63`, `etf_k70_63`, `etf_k90_63`, `etf_pc1_63`, `etf_absorption_63`, `etf_avg_corr_63`, `etf_mp_share_63`, `etf_dispersion_63`, `etf_n_kept_63`, `etf_eff_n_126`, `etf_k50_126`, `etf_k70_126`, `etf_k90_126`, `etf_pc1_126`, `etf_absorption_126`, `etf_avg_corr_126`, `etf_mp_share_126`, `etf_dispersion_126`, `etf_n_kept_126`, `etf_eff_n_252`, `etf_k50_252`, `etf_k70_252`, `etf_k90_252`, `etf_pc1_252`, `etf_absorption_252`, `etf_avg_corr_252`, `etf_mp_share_252`, `etf_dispersion_252`, `etf_n_kept_252`, `etf_eff_n_63_ratio_l5`, `etf_eff_n_63_ratio_l20`, `etf_k70_63_ratio_l5`, `etf_k70_63_ratio_l20`, `etf_pc1_63_ratio_l5`, `etf_pc1_63_ratio_l20`, `etf_avg_corr_63_ratio_l5`, `etf_avg_corr_63_ratio_l20`, `etf_eff_n_63_over_252`, `etf_avg_corr_63_over_252`, `etf_absorption_shift`

## pca_stock

`stock_eff_n_90`, `stock_k50_90`, `stock_k70_90`, `stock_k90_90`, `stock_pc1_90`, `stock_absorption_90`, `stock_avg_corr_90`, `stock_mp_share_90`, `stock_dispersion_90`, `stock_n_kept_90`, `stock_eff_n_250`, `stock_k50_250`, `stock_k70_250`, `stock_k90_250`, `stock_pc1_250`, `stock_absorption_250`, `stock_avg_corr_250`, `stock_mp_share_250`, `stock_dispersion_250`, `stock_n_kept_250`, `stock_eff_n_90_ratio_l5`, `stock_eff_n_90_ratio_l20`, `stock_k70_90_ratio_l5`, `stock_k70_90_ratio_l20`, `stock_pc1_90_ratio_l5`, `stock_pc1_90_ratio_l20`, `stock_avg_corr_90_ratio_l5`, `stock_avg_corr_90_ratio_l20`, `stock_eff_n_90_over_250`, `stock_avg_corr_90_over_250`, `stock_absorption_shift`, `stock_eff_n_90_tercile`

## macro

`rate_10y_level`, `curve_10y_3m`, `curve_10y_2y`, `rate_10y_change_21`, `curve_10y_3m_change_63`, `rate_3m_level`, `vix_level`, `vix_z_252`, `vix_change_5`, `vxn_over_vix`, `vix_over_vix3m`, `vix9d_over_vix`, `vvix_z_252`, `skew_z_252`, `cor1m_level`, `cor3m_level`, `cor1m_minus_cor3m`, `cor3m_change_5`, `credit_hyg_minus_lqd_5`, `rates_ief_minus_shy_5`, `credit_hyg_minus_lqd_21`, `rates_ief_minus_shy_21`, `dollar_uup_63`, `copper_over_gold_63`, `oil_wti_21`, `spy_above_ma200`, `spy_return_21`, `spy_vol_21`, `regime_composite`

## calendar

`day_of_week`, `month_sin`, `month_cos`, `business_days_to_month_end`, `turn_of_month`

# Model comparison

Metrics on the test users (19,682 users). Every setting was chosen on the validation users; the final model has the best validation F1.

| System | Setting (chosen on valid) | Valid F1 | Test F1 [95% CI] | Test precision@10 | Test recall@10 | Test ROC AUC | Avg items |
|---|---|---|---|---|---|---|---|
| repeat_last_order | products of the last order | 0.3138 | 0.312 [0.309, 0.316] | 0.230 | 0.455 | - | 10.3 |
| purchase_rate | order rate >= 0.26 | 0.3290 | 0.327 [0.324, 0.331] | 0.272 | 0.518 | 0.772 | 10.7 |
| logistic_regression | probability >= 0.18 | 0.3647 | 0.364 [0.360, 0.367] | 0.300 | 0.575 | 0.825 | 9.5 |
| random_forest | probability >= 0.20 | 0.3775 | 0.376 [0.372, 0.380] | 0.306 | 0.584 | 0.832 | 8.8 |
| **lightgbm** (final) | probability >= 0.21 | 0.3784 | 0.376 [0.373, 0.380] | 0.307 | 0.586 | 0.833 | 8.4 |

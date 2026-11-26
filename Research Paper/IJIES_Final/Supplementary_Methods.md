# Supplementary methods and predictor specification

## BIO_COLS

LBDHDD, LBXTC, NON_HDL_CHOL, CHOL_HDL_RATIO, NON_HDL_TO_HDL_RATIO.

## DIET_COLS

DR1TKCAL, DR1TPROT, DR1TCARB, DR1TSUGR, DR1TFIBE, DR1TTFAT, DR1TSFAT, DR1TMFAT, DR1TPFAT, DR1TCHOL, DR1TATOC, DR1TRET, DR1TVARA, DR1TACAR, DR1TBCAR, DR1TLYCO, DR1TLZ, DR1TVB1, DR1TVB2, DR1TNIAC, DR1TVB6, DR1TFOLA, DR1TFA, DR1TFF, DR1TFDFE, DR1TVB12, DR1TVC, DR1TVK, DR1TCALC, DR1TPHOS, DR1TMAGN, DR1TIRON, DR1TZINC, DR1TCOPP, DR1TSODI, DR1TPOTA, DR1TSELE, DR1TCAFF, DR1TALCO, PROT_PCT_KCAL, CARB_PCT_KCAL, FAT_PCT_KCAL, FIBER_PER_1000_KCAL, SUGAR_TO_FIBER_RATIO, SODIUM_POTASSIUM_RATIO, SAT_TO_UNSAT_FAT_RATIO.

## LIFE_COLS

SLEEP_HOURS, SLEEP_DEV_FROM_7_5, PAD680, PA_VIG_WORK_MIN_WK, PA_MOD_WORK_MIN_WK, PA_WALK_BIKE_MIN_WK, PA_VIG_REC_MIN_WK, PA_MOD_REC_MIN_WK, PA_TOTAL_MET_MIN_WK, SLQ050, PAQ605, PAQ620, PAQ635, PAQ650, PAQ665, SMOKE_NEVER, SMOKE_FORMER, SMOKE_SOMEDAYS, SMOKE_EVERYDAY.

## DEMO_COLS

RIDAGEYR, INDFMPIR, DMDEDUC2, SEX_FEMALE, RACE_MEX_AMER, RACE_OTHER_HISP, RACE_NH_WHITE, RACE_NH_BLACK, RACE_OTHER_MULTI, MARITAL_MARRIED, MARITAL_PREV_MARRIED, MARITAL_NEVER_MARRIED, INDFMPIR_MISS, SLEEP_HOURS_MISS, PAD680_MISS, PA_TOTAL_MET_MIN_WK_MISS, OTHER_FEATURE_MISS.


## Model selection configurations

All families use the same partitions and four predefined configurations. Selection is based on validation mean diabetes and CVD AUROC; calibration and temporal outcomes do not select configurations.

### logistic_regression

1. C = 0.01.

2. C = 0.1.

3. C = 1.0.

4. C = 10.0.

### hist_gradient_boosting

1. learning_rate = 0.05, max_leaf_nodes = 15, l2_regularization = 1.0, max_iter = 300.

2. learning_rate = 0.1, max_leaf_nodes = 15, l2_regularization = 1.0, max_iter = 300.

3. learning_rate = 0.05, max_leaf_nodes = 31, l2_regularization = 1.0, max_iter = 300.

4. learning_rate = 0.1, max_leaf_nodes = 31, l2_regularization = 1.0, max_iter = 300.

### catboost

1. depth = 4, learning_rate = 0.03, iterations = 600, l2_leaf_reg = 3.0.

2. depth = 4, learning_rate = 0.1, iterations = 600, l2_leaf_reg = 3.0.

3. depth = 6, learning_rate = 0.03, iterations = 600, l2_leaf_reg = 3.0.

4. depth = 6, learning_rate = 0.1, iterations = 600, l2_leaf_reg = 3.0.

### mlp

1. hidden = [256, 128], dropout = 0.1, lr = 0.0003, weight_decay = 0.0001.

2. hidden = [256, 128], dropout = 0.1, lr = 0.001, weight_decay = 0.0001.

3. hidden = [256, 256, 128], dropout = 0.2, lr = 0.0003, weight_decay = 0.0001.

4. hidden = [256, 256, 128], dropout = 0.2, lr = 0.001, weight_decay = 0.0001.

### bionutrigpt

1. lr = 0.0001, weight_decay = 0.0001.

2. lr = 0.0003, weight_decay = 0.0001.

3. lr = 0.0001, weight_decay = 0.001.

4. lr = 0.0003, weight_decay = 0.001.

### ft_transformer

1. n_blocks = 2, lr = 0.0001, weight_decay = 0.0001.

2. n_blocks = 2, lr = 0.0003, weight_decay = 0.0001.

3. n_blocks = 3, lr = 0.0001, weight_decay = 0.0001.

4. n_blocks = 3, lr = 0.0003, weight_decay = 0.0001.


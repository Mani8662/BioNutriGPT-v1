# Stage 1 Demographic Subgroup Evaluation Report (v1)

**Generated UTC:** 2026-11-20 12:23:42
**Scope:** N = 7,004 temporal evaluation participants; six frozen model families; three clinical outcomes (Diabetes, CVD primary; Obesity exploratory).
**Analysis Policy:** Analysis plan fixed after inspection of earlier overall temporal results; not a prospective preregistration. Unweighted conditional-on-fitted-model analysis.

---

## 1. Executive Summary & Key Observations

The six frozen models were evaluated on the same 7,004 temporal participants. Results are unweighted and conditional on the fitted models. They do not establish clinical fairness, Indian generalizability, or causes of subgroup differences.

Across age strata, calibrated diabetes AUROC ranges from 0.706 to 0.778 and CVD AUROC from 0.710 to 0.764. For ages 20–40, BioNutriGPT diabetes AUROC is 0.7562. The previous summary's 0.82–0.85 range for this age group was incorrect.

Of 120 calibrated diabetes/CVD AUROC and Brier contrasts against CatBoost, histogram boosting and FT-Transformer, 12 pointwise 95% intervals exclude zero. Directions are mixed. These are exploratory comparisons without multiplicity adjustment; no overall architectural advantage or model equivalence is established.

Subgroup disease prevalence differs, but this analysis does not establish that prevalence explains performance differences. No group triggers the pragmatic N<100 or positive/negative<20 warning; this does not guarantee adequate precision (young-adult CVD has only 37 positive cases).

The ranges below are point estimates across six models, not confidence intervals.

| Outcome | Demographic dimension | Minimum AUROC | Maximum AUROC |
|---|---|---:|---:|
| CVD | Age | 0.7102 | 0.7641 |
| CVD | Race_Ethnicity | 0.7729 | 0.8771 |
| CVD | Sex | 0.8085 | 0.8378 |
| DIABETES | Age | 0.7061 | 0.7777 |
| DIABETES | Race_Ethnicity | 0.7823 | 0.8313 |
| DIABETES | Sex | 0.7829 | 0.8201 |

---

## 2. Demographic Subgroup Counts and Prevalence

| Demographic Dimension | Subgroup | N | Diabetes Pos (%) | Obesity Pos (%) | CVD Pos (%) | Small-Group Flag |
|---|---|---:|---:|---:|---:|---|
| Sex | Female | 3,599 | 603 (16.8%) | 1679 (46.7%) | 346 (9.6%) | None |
| Sex | Male | 3,405 | 698 (20.5%) | 1378 (40.5%) | 499 (14.7%) | None |
| Age | 20-40 | 2,257 | 86 (3.8%) | 930 (41.2%) | 37 (1.6%) | None |
| Age | 41-65 | 3,173 | 699 (22.0%) | 1491 (47.0%) | 359 (11.3%) | None |
| Age | >65 | 1,574 | 516 (32.8%) | 636 (40.4%) | 449 (28.5%) | None |
| Race_Ethnicity | Mexican American | 840 | 182 (21.7%) | 428 (51.0%) | 60 (7.1%) | None |
| Race_Ethnicity | Other Hispanic | 723 | 136 (18.8%) | 309 (42.7%) | 67 (9.3%) | None |
| Race_Ethnicity | Non-Hispanic White | 2,564 | 408 (15.9%) | 1104 (43.1%) | 425 (16.6%) | None |
| Race_Ethnicity | Non-Hispanic Black | 1,784 | 374 (21.0%) | 937 (52.5%) | 210 (11.8%) | None |
| Race_Ethnicity | Other Race - Including Multi-Racial | 1,093 | 201 (18.4%) | 279 (25.5%) | 83 (7.6%) | None |

---

## 3. Primary Calibrated Performance: Diabetes

| Subgroup | Model | Calibrated AUROC [95% CI] | Calibrated Brier [95% CI] | Calibration in the Large | Joint Slope |
|---|---|:---:|:---:|:---:|:---:|
| Female | `logistic_regression` | 0.7829 [0.7638, 0.8014] | 0.1190 [0.1118, 0.1265] | +0.0340 | 0.9657 |
| Female | `hist_gradient_boosting` | 0.8042 [0.7863, 0.8222] | 0.1154 [0.1084, 0.1225] | +0.0475 | 0.9311 |
| Female | `catboost` | 0.8053 [0.7873, 0.8226] | 0.1150 [0.1080, 0.1221] | +0.0653 | 0.9406 |
| Female | `mlp` | 0.7911 [0.7733, 0.8089] | 0.1183 [0.1109, 0.1255] | +0.0586 | 0.9116 |
| Female | `bionutrigpt` | 0.8021 [0.7836, 0.8199] | 0.1155 [0.1090, 0.1225] | +0.0062 | 0.9584 |
| Female | `ft_transformer` | 0.7998 [0.7829, 0.8175] | 0.1163 [0.1096, 0.1233] | +0.0063 | 0.9263 |
| Male | `logistic_regression` | 0.8069 [0.7893, 0.8232] | 0.1291 [0.1217, 0.1366] | +0.1421 | 1.0311 |
| Male | `hist_gradient_boosting` | 0.8179 [0.8019, 0.8338] | 0.1262 [0.1193, 0.1331] | +0.1018 | 1.0152 |
| Male | `catboost` | 0.8194 [0.8026, 0.8352] | 0.1255 [0.1189, 0.1325] | +0.0718 | 1.0172 |
| Male | `mlp` | 0.8061 [0.7886, 0.8231] | 0.1284 [0.1211, 0.1360] | +0.1491 | 0.9706 |
| Male | `bionutrigpt` | 0.8169 [0.8000, 0.8327] | 0.1267 [0.1199, 0.1332] | +0.0938 | 1.0404 |
| Male | `ft_transformer` | 0.8201 [0.8038, 0.8355] | 0.1254 [0.1185, 0.1325] | +0.1352 | 1.0263 |
| 20-40 | `logistic_regression` | 0.7061 [0.6460, 0.7641] | 0.0369 [0.0312, 0.0436] | -0.6093 | 0.8533 |
| 20-40 | `hist_gradient_boosting` | 0.7547 [0.7020, 0.8037] | 0.0349 [0.0287, 0.0419] | -0.0660 | 1.2278 |
| 20-40 | `catboost` | 0.7665 [0.7189, 0.8138] | 0.0351 [0.0289, 0.0424] | +0.1201 | 1.1405 |
| 20-40 | `mlp` | 0.7518 [0.7003, 0.8026] | 0.0354 [0.0294, 0.0420] | -0.4119 | 1.0520 |
| 20-40 | `bionutrigpt` | 0.7562 [0.7039, 0.8054] | 0.0351 [0.0288, 0.0423] | +0.1231 | 1.0947 |
| 20-40 | `ft_transformer` | 0.7777 [0.7299, 0.8230] | 0.0345 [0.0285, 0.0413] | -0.1835 | 1.1038 |
| 41-65 | `logistic_regression` | 0.7290 [0.7084, 0.7491] | 0.1536 [0.1453, 0.1621] | +0.2909 | 0.8874 |
| 41-65 | `hist_gradient_boosting` | 0.7445 [0.7236, 0.7641] | 0.1492 [0.1418, 0.1571] | +0.0447 | 0.9013 |
| 41-65 | `catboost` | 0.7455 [0.7253, 0.7653] | 0.1487 [0.1410, 0.1566] | +0.0454 | 0.9593 |
| 41-65 | `mlp` | 0.7253 [0.7043, 0.7463] | 0.1533 [0.1449, 0.1620] | +0.2436 | 0.7837 |
| 41-65 | `bionutrigpt` | 0.7353 [0.7145, 0.7560] | 0.1502 [0.1430, 0.1578] | +0.0197 | 0.9537 |
| 41-65 | `ft_transformer` | 0.7361 [0.7153, 0.7560] | 0.1499 [0.1424, 0.1579] | +0.0654 | 0.9016 |
| >65 | `logistic_regression` | 0.7332 [0.7080, 0.7583] | 0.1886 [0.1791, 0.1982] | +0.0386 | 0.9633 |
| >65 | `hist_gradient_boosting` | 0.7420 [0.7171, 0.7677] | 0.1860 [0.1752, 0.1963] | +0.1698 | 0.8857 |
| >65 | `catboost` | 0.7447 [0.7193, 0.7698] | 0.1845 [0.1741, 0.1946] | +0.0925 | 0.9637 |
| >65 | `mlp` | 0.7314 [0.7052, 0.7572] | 0.1884 [0.1786, 0.1982] | +0.0720 | 0.9166 |
| >65 | `bionutrigpt` | 0.7472 [0.7225, 0.7716] | 0.1849 [0.1755, 0.1939] | +0.0829 | 1.1116 |
| >65 | `ft_transformer` | 0.7445 [0.7196, 0.7704] | 0.1856 [0.1752, 0.1963] | +0.1638 | 0.8626 |
| Mexican American | `logistic_regression` | 0.8003 [0.7656, 0.8331] | 0.1362 [0.1210, 0.1516] | +0.2390 | 1.0845 |
| Mexican American | `hist_gradient_boosting` | 0.8111 [0.7777, 0.8434] | 0.1331 [0.1178, 0.1484] | +0.2479 | 0.9498 |
| Mexican American | `catboost` | 0.8177 [0.7852, 0.8491] | 0.1315 [0.1162, 0.1470] | +0.2952 | 0.9988 |
| Mexican American | `mlp` | 0.8130 [0.7809, 0.8448] | 0.1329 [0.1179, 0.1477] | +0.1731 | 0.9876 |
| Mexican American | `bionutrigpt` | 0.8081 [0.7756, 0.8404] | 0.1344 [0.1198, 0.1489] | +0.2497 | 0.9223 |
| Mexican American | `ft_transformer` | 0.8193 [0.7881, 0.8510] | 0.1294 [0.1156, 0.1431] | +0.0930 | 1.0615 |
| Other Hispanic | `logistic_regression` | 0.7823 [0.7446, 0.8219] | 0.1299 [0.1151, 0.1459] | +0.0410 | 0.9858 |
| Other Hispanic | `hist_gradient_boosting` | 0.8016 [0.7663, 0.8376] | 0.1258 [0.1113, 0.1409] | -0.0445 | 0.9566 |
| Other Hispanic | `catboost` | 0.8031 [0.7657, 0.8394] | 0.1249 [0.1111, 0.1399] | -0.0482 | 0.9709 |
| Other Hispanic | `mlp` | 0.7867 [0.7497, 0.8249] | 0.1293 [0.1148, 0.1452] | +0.0172 | 0.8734 |
| Other Hispanic | `bionutrigpt` | 0.8124 [0.7772, 0.8491] | 0.1230 [0.1097, 0.1374] | -0.0233 | 1.0418 |
| Other Hispanic | `ft_transformer` | 0.7998 [0.7631, 0.8371] | 0.1256 [0.1116, 0.1409] | -0.0649 | 0.9838 |
| Non-Hispanic White | `logistic_regression` | 0.8046 [0.7814, 0.8256] | 0.1104 [0.1022, 0.1187] | +0.1588 | 1.0608 |
| Non-Hispanic White | `hist_gradient_boosting` | 0.8187 [0.7967, 0.8395] | 0.1063 [0.0986, 0.1145] | +0.1002 | 1.0483 |
| Non-Hispanic White | `catboost` | 0.8197 [0.7976, 0.8407] | 0.1053 [0.0977, 0.1133] | +0.0902 | 1.0777 |
| Non-Hispanic White | `mlp` | 0.8025 [0.7791, 0.8242] | 0.1096 [0.1017, 0.1177] | +0.1036 | 0.9934 |
| Non-Hispanic White | `bionutrigpt` | 0.8225 [0.7999, 0.8423] | 0.1058 [0.0983, 0.1135] | +0.0960 | 1.0609 |
| Non-Hispanic White | `ft_transformer` | 0.8193 [0.7971, 0.8391] | 0.1079 [0.0996, 0.1165] | +0.3314 | 1.0060 |
| Non-Hispanic Black | `logistic_regression` | 0.7852 [0.7595, 0.8091] | 0.1373 [0.1284, 0.1469] | -0.1342 | 0.9527 |
| Non-Hispanic Black | `hist_gradient_boosting` | 0.7923 [0.7673, 0.8151] | 0.1375 [0.1287, 0.1476] | -0.0891 | 0.8838 |
| Non-Hispanic Black | `catboost` | 0.7934 [0.7696, 0.8154] | 0.1376 [0.1285, 0.1474] | -0.1112 | 0.8671 |
| Non-Hispanic Black | `mlp` | 0.7898 [0.7651, 0.8122] | 0.1364 [0.1272, 0.1463] | -0.0684 | 0.9748 |
| Non-Hispanic Black | `bionutrigpt` | 0.7908 [0.7670, 0.8146] | 0.1375 [0.1293, 0.1463] | -0.1896 | 1.0467 |
| Non-Hispanic Black | `ft_transformer` | 0.7904 [0.7669, 0.8134] | 0.1370 [0.1285, 0.1465] | -0.1077 | 0.9816 |
| Other Race - Including Multi-Racial | `logistic_regression` | 0.8005 [0.7657, 0.8314] | 0.1202 [0.1065, 0.1333] | +0.2683 | 1.1194 |
| Other Race - Including Multi-Racial | `hist_gradient_boosting` | 0.8313 [0.8006, 0.8574] | 0.1137 [0.1005, 0.1268] | +0.2791 | 1.1382 |
| Other Race - Including Multi-Racial | `catboost` | 0.8298 [0.7994, 0.8567] | 0.1144 [0.1015, 0.1270] | +0.2572 | 1.1216 |
| Other Race - Including Multi-Racial | `mlp` | 0.8101 [0.7788, 0.8383] | 0.1221 [0.1074, 0.1363] | +0.4800 | 0.9589 |
| Other Race - Including Multi-Racial | `bionutrigpt` | 0.8231 [0.7925, 0.8511] | 0.1177 [0.1042, 0.1306] | +0.3415 | 1.0793 |
| Other Race - Including Multi-Racial | `ft_transformer` | 0.8274 [0.7975, 0.8556] | 0.1143 [0.1022, 0.1264] | -0.0172 | 1.0801 |

---

## 4. Primary Calibrated Performance: Cardiovascular Disease (CVD)

| Subgroup | Model | Calibrated AUROC [95% CI] | Calibrated Brier [95% CI] | Calibration in the Large | Joint Slope |
|---|---|:---:|:---:|:---:|:---:|
| Female | `logistic_regression` | 0.8148 [0.7904, 0.8366] | 0.0745 [0.0682, 0.0811] | +0.0211 | 0.9495 |
| Female | `hist_gradient_boosting` | 0.8117 [0.7874, 0.8341] | 0.0751 [0.0687, 0.0815] | -0.0652 | 0.9651 |
| Female | `catboost` | 0.8134 [0.7893, 0.8356] | 0.0754 [0.0690, 0.0819] | -0.0985 | 0.9090 |
| Female | `mlp` | 0.8085 [0.7849, 0.8309] | 0.0751 [0.0688, 0.0816] | -0.0108 | 0.9394 |
| Female | `bionutrigpt` | 0.8130 [0.7896, 0.8349] | 0.0748 [0.0688, 0.0809] | -0.1378 | 0.9081 |
| Female | `ft_transformer` | 0.8203 [0.7972, 0.8420] | 0.0742 [0.0678, 0.0806] | +0.0162 | 0.9417 |
| Male | `logistic_regression` | 0.8314 [0.8144, 0.8479] | 0.1018 [0.0951, 0.1089] | +0.0976 | 0.9454 |
| Male | `hist_gradient_boosting` | 0.8274 [0.8109, 0.8438] | 0.1031 [0.0963, 0.1100] | +0.1037 | 0.9212 |
| Male | `catboost` | 0.8336 [0.8173, 0.8495] | 0.1013 [0.0946, 0.1083] | +0.0788 | 0.9140 |
| Male | `mlp` | 0.8329 [0.8159, 0.8490] | 0.1013 [0.0949, 0.1082] | +0.0627 | 0.9616 |
| Male | `bionutrigpt` | 0.8352 [0.8184, 0.8511] | 0.1006 [0.0942, 0.1075] | +0.1077 | 0.9256 |
| Male | `ft_transformer` | 0.8378 [0.8219, 0.8537] | 0.1001 [0.0936, 0.1070] | +0.0368 | 0.9461 |
| 20-40 | `logistic_regression` | 0.7543 [0.6755, 0.8301] | 0.0160 [0.0115, 0.0206] | -0.2966 | 0.9818 |
| 20-40 | `hist_gradient_boosting` | 0.7346 [0.6509, 0.8143] | 0.0158 [0.0113, 0.0205] | -0.1856 | 1.0420 |
| 20-40 | `catboost` | 0.7609 [0.6789, 0.8411] | 0.0157 [0.0111, 0.0205] | +0.1534 | 1.2899 |
| 20-40 | `mlp` | 0.7424 [0.6631, 0.8220] | 0.0157 [0.0113, 0.0204] | -0.2958 | 1.1389 |
| 20-40 | `bionutrigpt` | 0.7632 [0.6808, 0.8378] | 0.0158 [0.0113, 0.0206] | +0.1111 | 0.9674 |
| 20-40 | `ft_transformer` | 0.7641 [0.6836, 0.8382] | 0.0158 [0.0113, 0.0205] | -0.0897 | 1.0335 |
| 41-65 | `logistic_regression` | 0.7480 [0.7209, 0.7746] | 0.0928 [0.0846, 0.1005] | +0.1815 | 0.8845 |
| 41-65 | `hist_gradient_boosting` | 0.7385 [0.7110, 0.7647] | 0.0943 [0.0865, 0.1017] | +0.0355 | 0.8106 |
| 41-65 | `catboost` | 0.7473 [0.7216, 0.7723] | 0.0928 [0.0847, 0.1003] | +0.1118 | 0.9447 |
| 41-65 | `mlp` | 0.7469 [0.7198, 0.7739] | 0.0925 [0.0847, 0.1000] | +0.1153 | 0.8395 |
| 41-65 | `bionutrigpt` | 0.7457 [0.7189, 0.7720] | 0.0921 [0.0845, 0.0991] | -0.0155 | 0.8433 |
| 41-65 | `ft_transformer` | 0.7570 [0.7306, 0.7817] | 0.0920 [0.0843, 0.0995] | +0.0604 | 0.8905 |
| >65 | `logistic_regression` | 0.7183 [0.6899, 0.7440] | 0.1808 [0.1707, 0.1914] | +0.0033 | 0.9345 |
| >65 | `hist_gradient_boosting` | 0.7129 [0.6857, 0.7392] | 0.1820 [0.1718, 0.1927] | +0.0475 | 0.9596 |
| >65 | `catboost` | 0.7139 [0.6859, 0.7411] | 0.1820 [0.1726, 0.1915] | -0.1292 | 0.9676 |
| >65 | `mlp` | 0.7102 [0.6823, 0.7368] | 0.1819 [0.1726, 0.1918] | -0.0039 | 0.9728 |
| >65 | `bionutrigpt` | 0.7147 [0.6865, 0.7418] | 0.1804 [0.1709, 0.1904] | -0.0094 | 1.0020 |
| >65 | `ft_transformer` | 0.7259 [0.6975, 0.7523] | 0.1780 [0.1680, 0.1884] | +0.0104 | 0.9056 |
| Mexican American | `logistic_regression` | 0.8580 [0.8060, 0.9012] | 0.0562 [0.0447, 0.0687] | +0.1009 | 1.0969 |
| Mexican American | `hist_gradient_boosting` | 0.8619 [0.8150, 0.9003] | 0.0553 [0.0450, 0.0672] | -0.0887 | 1.1842 |
| Mexican American | `catboost` | 0.8771 [0.8375, 0.9111] | 0.0547 [0.0440, 0.0658] | -0.1716 | 1.1415 |
| Mexican American | `mlp` | 0.8710 [0.8237, 0.9097] | 0.0548 [0.0439, 0.0667] | +0.0313 | 1.2320 |
| Mexican American | `bionutrigpt` | 0.8764 [0.8343, 0.9113] | 0.0545 [0.0448, 0.0656] | -0.1985 | 1.1361 |
| Mexican American | `ft_transformer` | 0.8731 [0.8297, 0.9101] | 0.0546 [0.0439, 0.0664] | -0.0236 | 1.2689 |
| Other Hispanic | `logistic_regression` | 0.8100 [0.7571, 0.8597] | 0.0747 [0.0611, 0.0893] | -0.0482 | 0.9101 |
| Other Hispanic | `hist_gradient_boosting` | 0.8218 [0.7733, 0.8674] | 0.0728 [0.0598, 0.0866] | -0.1465 | 0.9738 |
| Other Hispanic | `catboost` | 0.8167 [0.7658, 0.8620] | 0.0737 [0.0608, 0.0874] | -0.2186 | 0.9040 |
| Other Hispanic | `mlp` | 0.7998 [0.7470, 0.8486] | 0.0757 [0.0620, 0.0907] | -0.0493 | 0.8509 |
| Other Hispanic | `bionutrigpt` | 0.8198 [0.7704, 0.8663] | 0.0735 [0.0610, 0.0874] | -0.1169 | 0.8964 |
| Other Hispanic | `ft_transformer` | 0.8201 [0.7695, 0.8676] | 0.0735 [0.0605, 0.0880] | -0.0514 | 0.9442 |
| Non-Hispanic White | `logistic_regression` | 0.8248 [0.8042, 0.8443] | 0.1102 [0.1023, 0.1177] | +0.1297 | 0.9964 |
| Non-Hispanic White | `hist_gradient_boosting` | 0.8220 [0.8004, 0.8409] | 0.1116 [0.1038, 0.1191] | +0.1409 | 0.9653 |
| Non-Hispanic White | `catboost` | 0.8243 [0.8037, 0.8432] | 0.1103 [0.1023, 0.1178] | +0.1063 | 0.9406 |
| Non-Hispanic White | `mlp` | 0.8225 [0.8015, 0.8422] | 0.1108 [0.1030, 0.1182] | +0.1221 | 0.9451 |
| Non-Hispanic White | `bionutrigpt` | 0.8214 [0.8009, 0.8413] | 0.1105 [0.1028, 0.1179] | +0.1196 | 0.9300 |
| Non-Hispanic White | `ft_transformer` | 0.8337 [0.8135, 0.8522] | 0.1088 [0.1011, 0.1163] | +0.1046 | 0.9054 |
| Non-Hispanic Black | `logistic_regression` | 0.7870 [0.7556, 0.8167] | 0.0927 [0.0828, 0.1025] | -0.0688 | 0.8162 |
| Non-Hispanic Black | `hist_gradient_boosting` | 0.7729 [0.7398, 0.8046] | 0.0954 [0.0852, 0.1055] | -0.0387 | 0.7547 |
| Non-Hispanic Black | `catboost` | 0.7782 [0.7458, 0.8086] | 0.0940 [0.0836, 0.1042] | -0.0222 | 0.7475 |
| Non-Hispanic Black | `mlp` | 0.7811 [0.7482, 0.8111] | 0.0930 [0.0833, 0.1026] | -0.1106 | 0.8665 |
| Non-Hispanic Black | `bionutrigpt` | 0.7852 [0.7529, 0.8157] | 0.0920 [0.0825, 0.1012] | -0.1746 | 0.8376 |
| Non-Hispanic Black | `ft_transformer` | 0.7879 [0.7563, 0.8172] | 0.0921 [0.0823, 0.1022] | -0.0576 | 0.8655 |
| Other Race - Including Multi-Racial | `logistic_regression` | 0.8532 [0.8161, 0.8871] | 0.0602 [0.0493, 0.0718] | +0.1713 | 1.0171 |
| Other Race - Including Multi-Racial | `hist_gradient_boosting` | 0.8495 [0.8125, 0.8828] | 0.0603 [0.0501, 0.0715] | -0.0591 | 1.0320 |
| Other Race - Including Multi-Racial | `catboost` | 0.8585 [0.8237, 0.8900] | 0.0609 [0.0506, 0.0717] | -0.0936 | 0.9804 |
| Other Race - Including Multi-Racial | `mlp` | 0.8551 [0.8169, 0.8906] | 0.0592 [0.0487, 0.0700] | +0.0612 | 1.0223 |
| Other Race - Including Multi-Racial | `bionutrigpt` | 0.8574 [0.8228, 0.8900] | 0.0599 [0.0493, 0.0711] | +0.1598 | 0.9549 |
| Other Race - Including Multi-Racial | `ft_transformer` | 0.8504 [0.8124, 0.8837] | 0.0599 [0.0494, 0.0712] | +0.0340 | 0.9997 |

---

## 5. Paired Differences: BioNutriGPT Minus Comparators (Calibrated)

Positive AUROC difference favors BioNutriGPT; negative Brier difference favors BioNutriGPT.

| Subgroup | Outcome | Comparator | $\Delta$ AUROC [95% CI] | $\Delta$ Brier [95% CI] |
|---|---|---|:---:|:---:|
| Female | Diabetes | `catboost` | -0.0032 [-0.0093, +0.0025] | +0.0005 [-0.0010, +0.0018] |
| Female | Diabetes | `ft_transformer` | +0.0023 [-0.0045, +0.0094] | -0.0009 [-0.0025, +0.0008] |
| Female | Diabetes | `logistic_regression` | +0.0192 [+0.0092, +0.0291] | -0.0035 [-0.0056, -0.0014] |
| Female | CVD | `catboost` | -0.0005 [-0.0088, +0.0081] | -0.0006 [-0.0019, +0.0006] |
| Female | CVD | `ft_transformer` | -0.0073 [-0.0143, -0.0011] | +0.0006 [-0.0006, +0.0017] |
| Female | CVD | `logistic_regression` | -0.0019 [-0.0106, +0.0073] | +0.0002 [-0.0013, +0.0017] |
| Male | Diabetes | `catboost` | -0.0026 [-0.0079, +0.0030] | +0.0011 [-0.0005, +0.0027] |
| Male | Diabetes | `ft_transformer` | -0.0032 [-0.0106, +0.0035] | +0.0013 [-0.0007, +0.0034] |
| Male | Diabetes | `logistic_regression` | +0.0099 [+0.0020, +0.0184] | -0.0024 [-0.0048, -0.0001] |
| Male | CVD | `catboost` | +0.0016 [-0.0044, +0.0073] | -0.0007 [-0.0023, +0.0010] |
| Male | CVD | `ft_transformer` | -0.0027 [-0.0080, +0.0027] | +0.0006 [-0.0009, +0.0021] |
| Male | CVD | `logistic_regression` | +0.0038 [-0.0033, +0.0108] | -0.0012 [-0.0032, +0.0008] |
| 20-40 | Diabetes | `catboost` | -0.0103 [-0.0448, +0.0222] | +0.0000 [-0.0006, +0.0007] |
| 20-40 | Diabetes | `ft_transformer` | -0.0214 [-0.0499, +0.0049] | +0.0006 [-0.0002, +0.0015] |
| 20-40 | Diabetes | `logistic_regression` | +0.0501 [+0.0144, +0.0883] | -0.0018 [-0.0029, -0.0005] |
| 20-40 | CVD | `catboost` | +0.0023 [-0.0399, +0.0409] | +0.0001 [-0.0001, +0.0003] |
| 20-40 | CVD | `ft_transformer` | -0.0010 [-0.0310, +0.0295] | +0.0001 [-0.0001, +0.0003] |
| 20-40 | CVD | `logistic_regression` | +0.0088 [-0.0307, +0.0473] | -0.0001 [-0.0004, +0.0002] |
| 41-65 | Diabetes | `catboost` | -0.0102 [-0.0189, -0.0018] | +0.0015 [-0.0003, +0.0034] |
| 41-65 | Diabetes | `ft_transformer` | -0.0007 [-0.0111, +0.0094] | +0.0004 [-0.0016, +0.0024] |
| 41-65 | Diabetes | `logistic_regression` | +0.0063 [-0.0046, +0.0172] | -0.0034 [-0.0060, -0.0007] |
| 41-65 | CVD | `catboost` | -0.0016 [-0.0139, +0.0108] | -0.0007 [-0.0020, +0.0008] |
| 41-65 | CVD | `ft_transformer` | -0.0113 [-0.0223, -0.0009] | +0.0001 [-0.0012, +0.0014] |
| 41-65 | CVD | `logistic_regression` | -0.0022 [-0.0151, +0.0110] | -0.0007 [-0.0023, +0.0009] |
| >65 | Diabetes | `catboost` | +0.0025 [-0.0064, +0.0118] | +0.0004 [-0.0027, +0.0033] |
| >65 | Diabetes | `ft_transformer` | +0.0026 [-0.0087, +0.0146] | -0.0007 [-0.0045, +0.0031] |
| >65 | Diabetes | `logistic_regression` | +0.0140 [+0.0001, +0.0290] | -0.0037 [-0.0082, +0.0004] |
| >65 | CVD | `catboost` | +0.0009 [-0.0125, +0.0144] | -0.0017 [-0.0052, +0.0019] |
| >65 | CVD | `ft_transformer` | -0.0112 [-0.0233, +0.0020] | +0.0023 [-0.0012, +0.0056] |
| >65 | CVD | `logistic_regression` | -0.0036 [-0.0208, +0.0146] | -0.0005 [-0.0048, +0.0038] |
| Mexican American | Diabetes | `catboost` | -0.0096 [-0.0203, +0.0014] | +0.0029 [-0.0007, +0.0063] |
| Mexican American | Diabetes | `ft_transformer` | -0.0113 [-0.0222, +0.0002] | +0.0050 [+0.0012, +0.0086] |
| Mexican American | Diabetes | `logistic_regression` | +0.0078 [-0.0098, +0.0249] | -0.0018 [-0.0066, +0.0026] |
| Mexican American | CVD | `catboost` | -0.0008 [-0.0176, +0.0149] | -0.0001 [-0.0021, +0.0017] |
| Mexican American | CVD | `ft_transformer` | +0.0032 [-0.0118, +0.0185] | -0.0001 [-0.0024, +0.0022] |
| Mexican American | CVD | `logistic_regression` | +0.0183 [-0.0001, +0.0374] | -0.0017 [-0.0047, +0.0010] |
| Other Hispanic | Diabetes | `catboost` | +0.0093 [-0.0024, +0.0218] | -0.0019 [-0.0054, +0.0013] |
| Other Hispanic | Diabetes | `ft_transformer` | +0.0126 [-0.0021, +0.0272] | -0.0026 [-0.0063, +0.0011] |
| Other Hispanic | Diabetes | `logistic_regression` | +0.0301 [+0.0115, +0.0495] | -0.0069 [-0.0122, -0.0020] |
| Other Hispanic | CVD | `catboost` | +0.0031 [-0.0164, +0.0210] | -0.0001 [-0.0032, +0.0028] |
| Other Hispanic | CVD | `ft_transformer` | -0.0003 [-0.0152, +0.0143] | -0.0000 [-0.0023, +0.0023] |
| Other Hispanic | CVD | `logistic_regression` | +0.0099 [-0.0105, +0.0303] | -0.0012 [-0.0043, +0.0019] |
| Non-Hispanic White | Diabetes | `catboost` | +0.0027 [-0.0040, +0.0094] | +0.0004 [-0.0010, +0.0019] |
| Non-Hispanic White | Diabetes | `ft_transformer` | +0.0031 [-0.0039, +0.0102] | -0.0021 [-0.0041, -0.0001] |
| Non-Hispanic White | Diabetes | `logistic_regression` | +0.0178 [+0.0064, +0.0298] | -0.0046 [-0.0072, -0.0021] |
| Non-Hispanic White | CVD | `catboost` | -0.0029 [-0.0098, +0.0040] | +0.0002 [-0.0018, +0.0021] |
| Non-Hispanic White | CVD | `ft_transformer` | -0.0123 [-0.0187, -0.0057] | +0.0017 [-0.0003, +0.0036] |
| Non-Hispanic White | CVD | `logistic_regression` | -0.0034 [-0.0123, +0.0056] | +0.0003 [-0.0022, +0.0029] |
| Non-Hispanic Black | Diabetes | `catboost` | -0.0026 [-0.0103, +0.0050] | -0.0002 [-0.0028, +0.0026] |
| Non-Hispanic Black | Diabetes | `ft_transformer` | +0.0003 [-0.0090, +0.0100] | +0.0005 [-0.0023, +0.0031] |
| Non-Hispanic Black | Diabetes | `logistic_regression` | +0.0056 [-0.0064, +0.0177] | +0.0002 [-0.0034, +0.0037] |
| Non-Hispanic Black | CVD | `catboost` | +0.0070 [-0.0035, +0.0177] | -0.0020 [-0.0041, +0.0000] |
| Non-Hispanic Black | CVD | `ft_transformer` | -0.0028 [-0.0134, +0.0075] | -0.0001 [-0.0021, +0.0018] |
| Non-Hispanic Black | CVD | `logistic_regression` | -0.0018 [-0.0141, +0.0096] | -0.0007 [-0.0030, +0.0016] |
| Other Race - Including Multi-Racial | Diabetes | `catboost` | -0.0067 [-0.0194, +0.0054] | +0.0033 [+0.0007, +0.0059] |
| Other Race - Including Multi-Racial | Diabetes | `ft_transformer` | -0.0044 [-0.0176, +0.0091] | +0.0033 [-0.0008, +0.0074] |
| Other Race - Including Multi-Racial | Diabetes | `logistic_regression` | +0.0225 [+0.0067, +0.0399] | -0.0026 [-0.0061, +0.0007] |
| Other Race - Including Multi-Racial | CVD | `catboost` | -0.0011 [-0.0149, +0.0129] | -0.0009 [-0.0030, +0.0011] |
| Other Race - Including Multi-Racial | CVD | `ft_transformer` | +0.0070 [-0.0051, +0.0196] | -0.0000 [-0.0015, +0.0015] |
| Other Race - Including Multi-Racial | CVD | `logistic_regression` | +0.0042 [-0.0112, +0.0187] | -0.0003 [-0.0024, +0.0018] |

---

## 6. Methodological Limits & Survey Design Boundary

- **Unweighted Sample Evaluation:** These metrics reflect unweighted performance on the selected analytic sample ($N=7,004$) and are **not** nationally representative US population estimates.
- **Complex Survey Design:** Participant-level bootstrap estimates do not incorporate NHANES multi-stage clustering, stratification (SDMVSTRA/SDMVPSU), or sampling weights (WTMEC2YR / WTDRD1). A formal complex-survey domain analysis remains separate future work.
- **Top-Coding:** Age top-coding at 80 years restricts granular evaluation in the oldest-old population.
- **Non-Causal Interpretation:** The analysis describes performance differences; it does not determine their causes or establish clinical fairness.
- **No General Superiority:** No overall BioNutriGPT advantage is established. Some exploratory intervals exclude zero in either direction; this does not establish equivalence or adjusted subgroup superiority.

## Verification correction — 2026-11-20

Extended independent verification reproduced all 360 calibration rows, 360 model bootstrap intervals and 300 paired intervals using 2,000 shared participant resamples. Demographic values and group assignments were checked against source metadata. The original shuffled-ID test was repaired. Numerical result files are unchanged. See `Saved Model/stage1_subgroups_v1/extended_verification.json` and `verify_stage1_subgroups_extended.py`. This supplements the original metric checks; survey-design uncertainty remains unassessed.

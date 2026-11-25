# Catalog Wording and Counts Reconciliation Receipt

**Date:** 2026-11-21  
**Study Scope:** Stage 1 Downstream Added-Value Controlled Study  
**Bound Catalog File:** `food_composition_cooked_sourced.json`  
**Catalog File SHA-256:** `a66b145f088f02c82ed9e8f81639a063bf3aed0bc44fcb2aab381e8bda24e64b`  

---

## 1. Verified Catalog Structure and Counts

Inspection of the catalog bound to the downstream study reveals the following exact accounting:

| Dimension | Count | Description and Provenance |
| :--- | :---: | :--- |
| **Total Loaded Catalog Entries** | **49** | Complete JSON items loaded into memory by `s3.load_food_database`. <br>• **20 INDB Published Recipe Variants:** Indian Nutrient Databank (repository: Lindsay Jaacks INDB; e.g., Dal Tadka, Chana Masala, Rajma, Phulka, Khichdi, Idli, Dosa, Pulao). <br>• **29 USDA SR Legacy Single-Food Proxies:** Borrowed generic single ready-to-eat foods (e.g., fresh fruits, nuts, seeds, milks, eggs, chicken breast, salmon, sardines). |
| **Excluded for Missing Active Data** | **6** | Foods automatically excluded by Stage 3 solver because they lack complete data across the 30 active planning nutrients. All 6 are INDB recipes lacking micronutrients: `recipe_sambar_vegetable` (missing FA, FDFE), `recipe_panchmel_dal` (missing ATOC, FA, FDFE), `recipe_veg_pulao` (missing ATOC), `recipe_vegetable_poha` (missing RET, FA, FDFE, VB12), `recipe_methi_aloo` (missing FA, FDFE), `recipe_palak_paneer` (missing FA, FDFE, VB12). |
| **Mode-Eligible: `veg_nonveg_combo`** | **43** | 49 loaded minus 6 excluded for missing active data (14 INDB recipes + 29 USDA proxies, including 6 non-veg proteins). |
| **Mode-Eligible: `pure_non_vegetarian`** | **43** | 49 loaded minus 6 excluded for missing active data (14 INDB recipes + 29 USDA proxies, including 6 non-veg proteins). |
| **Mode-Eligible: `pure_vegetarian`** | **37** | 49 loaded minus 6 excluded for missing active data minus 6 non-vegetarian animal proteins (14 INDB recipes + 23 USDA vegetarian proxies). |

---

## 2. Historical Provenance of Conflicting Counts

- **"44 dishes":** Historical synthetic composite recipe catalog (from early project iterations with 44 estimated dishes). This catalog had unverified nutrient estimations and was retired. Wording in downstream reports loosely referring to a "44-dish Indian composite catalog" conflated this retired catalog with the active bound catalog.
- **"29 dishes":** Earlier intermediate sourced catalog version (comprising 20 INDB recipes + 9 initial USDA proxies). This was subsequently expanded to 49 items by incorporating additional single-food proxies.
- **"49 dishes":** The actual frozen catalog bound to the 4,800-case study in `binding.json`.
- **"Indian Clinical Validity Claim":** The catalog contains 20 INDB recipe variants and 29 generic USDA SR Legacy single-food proxies. It must NOT be described as an "independently validated 44-dish Indian composite catalog".

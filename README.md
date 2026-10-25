# Sri Lanka Household Electricity Risk

This project predicts whether a Sri Lankan household may use more than 90 kWh of electricity in the following month. It uses the real RECON-SL monthly consumption records and Wave 1 household survey. The output is a risk category, not an exact bill.

## Local problem and objective

Electricity bills are a concern for many households. Recent usage may rise without the household noticing the pattern. This project combines the previous three monthly readings with household and appliance information to give an early normal-risk or high-risk estimate.

This is binary classification:

- `0`: normal electricity consumption risk
- `1`: high electricity consumption risk, meaning the following month is above 90 kWh

## Dataset

The project uses the Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL):

- [Kaggle dataset](https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption)
- [Zenodo record 15023048](https://zenodo.org/records/15023048)
- [Official LIRNEasia repository](https://github.com/LIRNEasia/SL_electricity_consumption)
- [RECON-SL dataset paper](https://arxiv.org/abs/2609.28783)

The raw monthly file has 101,575 records, 4,063 households and 25 months from October 2022 to October 2024. After requiring three past readings, a next-month target and survey information that was already available, 26,937 household-months from 3,776 households were used.

Selected raw files:

- `monthly_consumption.csv`
- `survey_dates.csv`
- `w1_household_information_and_history.csv`
- `w1_appliances.csv`
- `w1_ac_roster.csv`
- `w1_electricity_generation_water_heating_cooking.csv`
- the official data dictionaries and table structure

Raw data is ignored by Git because household data and large source files should not be uploaded to a public repository.

## Target and features

The next reading is created within each household using a forward shift. At the fixed 90 kWh threshold, 56.00% of training rows were high risk. Both classes were above 10%, so the understandable 90 kWh definition was retained. The training-only 75th percentile was 156 kWh and would make only 24.75% of training rows high risk.

Main input groups:

- previous one, two and three monthly readings
- past-only mean, minimum, maximum, change and variation
- calendar month and quarter
- household size, floor area and survey economic group
- service area, housing type and meter type
- appliance counts, weekly use, AC use and appliance ownership
- cooking, water heating, solar generation and home-business fields

`household_ID` is never a model feature.

## Time split

| Split | Months | Records | High risk |
|---|---|---:|---:|
| Training | Dec 2023–Jun 2024 | 16,765 | 56.00% |
| Validation | Jul–Aug 2024 | 6,413 | 57.37% |
| Testing | Sep 2024 | 3,759 | 59.16% |

No calendar month appears in more than one split. Survey answers are used only when Wave 1 was collected before the feature month. Imputation, scaling and encoding are fitted inside training pipelines.

## Models and result

The comparison includes a previous-month rule, Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost and a soft-voting ensemble.

Random Forest was selected using a score defined before testing: 30% balanced accuracy, 20% high-risk recall, 25% F1 and 25% PR-AUC. CatBoost was extremely close, but the test month was not used to change the earlier choice.

| Test model | Balanced accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0.9034** | 0.8921 | 0.9145 | 0.9632 | **0.9723** |
| CatBoost | 0.9015 | **0.9020** | **0.9156** | 0.9549 | 0.9595 |
| XGBoost | 0.8957 | 0.8683 | 0.9038 | 0.9620 | 0.9719 |
| Logistic Regression | 0.8885 | 0.8350 | 0.8906 | 0.9554 | 0.9685 |

The soft vote did not improve validation selection score or test balanced accuracy enough to justify extra complexity. Recent consumption features were the strongest model influences. SHAP is used for global and single-household explanations.


## Installation and use

Use Python 3.10 or a compatible version.

```bash
pip install -r requirements.txt
```

Place the selected RECON-SL files in the paths described above, then run the notebook from its first cell to its last cell. The executed notebook included here has 55 cells and no errors.

To rebuild all model outputs:

```bash
python train_project.py
```

To run the frontend:

```bash
streamlit run app.py
```

To run the tests:

```bash
pytest -q
```

Seven tests cover model and schema loading, valid inference, missing fields, unseen categories, negative values and probability bounds.



## Related work

- Ariyarathne, Jayatissa and De Silva studied Sri Lankan domestic electricity estimation using income, household size, linear models and Random Forest on 820 survey records. Their task predicts consumption, while this project predicts a next-month risk class using longitudinal RECON-SL data. [Journal of Science, University of Kelaniya paper](https://josuk.sljol.info/articles/8031/files/submission/proof/8031-1-28648-1-10-20210618.pdf)
- Peplinski et al. combined smart meter, climate, building and socioeconomic data for residential demand estimation. Their work supports combining household context with consumption history, but it uses a different country and prediction task. [Applied Energy, DOI 10.1016/j.apenergy.2023.122413](https://doi.org/10.1016/j.apenergy.2023.122413)
- A 2024 PLOS One study compared tree ensembles and explainable AI for residential building electricity forecasting. It supports model comparison and SHAP, while this project focuses on Sri Lankan next-month classification. [PLOS One, DOI 10.1371/journal.pone.0307654](https://doi.org/10.1371/journal.pone.0307654)

Scores from other studies are not compared directly because their datasets, target definitions and time periods are different.

## Ethics and limitations

- Names, phone numbers, addresses and account numbers are not collected.
- Survey economic information is sensitive and must not be used to punish or exclude a household.
- Performance is checked for service-area and socio economic groups with at least 50 test records; smaller groups are not displayed.
- RECON-SL represents selected LECO service areas, not every Sri Lankan household.
- The untouched test period is one latest usable month, which is a meaningful but short future window.
- Household behaviour, appliances, tariffs and weather can change.
- This is decision support, not an exact bill forecast or final authority.

Future work should add newer months, test unseen regions and households, connect safe smart meter summaries, monitor drift and review group performance regularly.

## Author and declaration

Author: Keerthana AA  
Student ID: 269131G

Assignment declaration: I confirm that I will review, understand and be able to explain the submitted code, analysis and conclusions. Dataset sources and related work are acknowledged above.

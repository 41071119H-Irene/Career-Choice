# 🎓 Labor Market Overflow: Higher Education & Self-Employment in East Asia (2013-2023)

This repository contains the full analytical pipeline for researching the **"Higher Education Overflow"** phenomenon in East Asian labor markets (Taiwan, Japan, and South Korea). By utilizing panel data from 2013 to 2023, this project explores how rapid tertiary education expansion drives high-skilled labor into self-employment as a structural "social buffer."

## 📌 Research Overview
The core objective is to validate the **Overflow Hypothesis**: In East Asian contexts, where traditional organizational roles (corporate/white-collar) are saturated, higher education expansion doesn't just lead to better employment—it pushes graduates into self-employment.

### Key Hypotheses:
1. **The Interaction Effect**: The positive correlation between Higher Education (HE) rates and self-employment is significantly stronger in East Asian countries compared to other OECD nations.
2. **The Social Buffer**: Self-employment serves as a flexible reservoir for high-skilled labor, particularly during external shocks like the COVID-19 pandemic and the 2023 Generative AI boom.

---

## 📊 Visualization Gallery
The script automatically generates 8 core research charts that build the empirical argument:

| ID | Chart Name | Research Insight |
| :--- | :--- | :--- |
| **Chart 2** | Global Quadrant Analysis (2022) | Categorizes countries into "Opportunity" vs. "Necessity" driven self-employment. |
| **Chart 3** | East Asia Trend Trajectory | Tracks the unique self-employment paths of TWN, JPN, and KOR from 2013-2023. |
| **Chart 4** | Taiwan Case Deep Dive | Visualizes the time-series correlation between HE enrollment and entrepreneurial activity (TEA). |
| **Chart 6** | Interaction Slope Analysis | Empirically demonstrates the steeper "overflow slope" in East Asia. |
| **Chart 7** | Pandemic Comparison (2019 vs 2023) | Analyzes how COVID-19 and AI have intensified the overflow mechanism. |
| **Chart 8** | 2023 Global Quadrant | **Latest Data**: Maps the labor market landscape in the post-pandemic/AI era. |

---

## 🛠️ Features
- **Robust Data Cleaning**: Automated handling of complex Excel headers, percentage symbols, and non-numeric outliers.
- **Missing Value Imputation**: Utilizes linear interpolation to fill gaps in multi-national panel data, ensuring smooth time-series analysis.
- **Econometric Modeling**: Performs OLS regression with interaction terms (`HE_Rate * Is_East_Asia`) to prove statistical significance ($p < 0.001$).
- **Automated Reporting**: Generates comprehensive CSV reports for model coefficients, data quality, and reliability statistics.


This repository contains the analytical pipeline and econometric visualizations for researching the **"Higher Education Overflow"** phenomenon. Focusing on East Asian labor markets (Taiwan, Japan, and South Korea),compare with OECD countries, this project explores how tertiary education expansion acts as a structural "social buffer," diverting high-skilled labor into self-employment when organizational roles are saturated.

## 📌 Research Overview

In rigid labor markets, higher education expansion does not always lead to traditional professional employment. This project validates the **"Pressure Relief Valve Hypothesis"**: self-employment serves as a vital structural outlet for human capital that the formal corporate sector cannot absorb.

### Core Hypotheses:
1.  **H1: Institutional Moderation**: The "Overflow Effect" (Higher Education Rate → Self-employment) is significantly steeper in East Asian institutional regimes compared to other OECD nations due to specific corporate and cultural structures.
2.  **H2: The Quality-Dynamic Paradox**: East Asian labor overflow is characterized by "Necessity-driven" motives and low Total Early-stage Entrepreneurial Activity (TEA), representing a structural mismatch of human capital.
3.  **H3: Technological Catalysis (AI Shock)**: External shocks, particularly the **2023 Generative AI boom**, have intensified this diversion, forcing a more rapid transition into non-traditional career paths for the highly educated.

---

## 📊 Visualization Gallery (AHRD Standards)

The pipeline automatically generates five core figures that form the empirical backbone of the study:

| Figure | Name | Analytical Insight |
| :--- | :--- | :--- |
| **Figure 1** | **Interaction Effect Plot** | Empirically demonstrates H1: The significantly steeper "overflow slope" in East Asian regimes ($p < 0.001$). |
| **Figure 2** | **Structural Overflow Quadrant** | Maps the paradox of high self-employment volume paired with low entrepreneurial dynamics and survival-based motives. |
| **Figure 3** | **Mechanism Snapshot (2023)** | Visualizes the "Low Unemployment vs. High Self-employment" equilibrium in the post-pandemic/AI era. |
| **Figure 4** | **AI Shock Comparison** | Contrasts 2019 vs. 2023 regression slopes to demonstrate the intensifying pressure of workplace automation. |
| **Figure 5** | **Intra-Regional Longitudinal** | Tracks the "Career Reservoir" effect in TWN, JPN, and KOR, highlighting Taiwan's unique labor market resilience. |

---

## 🛠️ Technical Implementation

### Data Science Workflow:
-   **Econometric Modeling**: Employs **Hierarchical Regression Analysis** to test moderating effects and measure model fit improvements ($\Delta R^2$).
-   **Data Harmonization**: Integrates multi-source panel data (OECD, GEM, World Bank) with automated cleaning for non-numeric outliers and complex header formatting.
-   **Time-Series Imputation**: Handles missing values through linear interpolation to ensure continuity across 10-year longitudinal datasets.

### Tech Stack:
-   **Analysis**: Python (Pandas, Statsmodels, NumPy)
-   **Visualization**: Matplotlib, Seaborn (Optimized for academic publication aesthetics)
-   **Reporting**: Automated CSV exports for model coefficients, p-values, and descriptive statistics.

---

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# ==========================================
# 1. 配置與環境設定
# ==========================================
FILE_PATH = 'data.xlsx'
OUTPUT_DIR = "research_final_complete_package"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定學術繪圖風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 2. 資料清理工具
# ==========================================
def clean_numeric(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('%', '').replace(',', '').strip()
        if x in ['-', '', '..', '...']: return np.nan
        try: return float(x)
        except: return np.nan
    return float(x)

def melt_to_long(df, id_col, value_name, start_y=2013, end_y=2023):
    df.columns = [str(c).strip() for c in df.columns]
    year_cols = [c for c in df.columns if c.isdigit() and start_y <= int(c) <= end_y]
    df_long = df.melt(id_vars=[id_col], value_vars=year_cols, var_name='Year', value_name=value_name)
    df_long['Year'] = df_long['Year'].astype(int)
    df_long[value_name] = df_long[value_name].apply(clean_numeric)
    return df_long

# ==========================================
# 3. 數據庫建立引擎
# ==========================================
def build_thesis_database():
    log("正在對齊 2013-2023 全球面板數據...")
    xls = pd.ExcelFile(FILE_PATH)
    
    df_he = melt_to_long(pd.read_excel(xls, 'School enrollment, tertiary % g'), 'Country Code', 'HE_Rate').rename(columns={'Country Code':'Code'})
    df_unemp = melt_to_long(pd.read_excel(xls, 'Unemployment, total (% of total'), 'Country Code', 'Unemployment_Rate').rename(columns={'Country Code':'Code'})
    df_tea = melt_to_long(pd.read_excel(xls, 'Percentage of 18-64 population ', header=1), 'Code', 'TEA')
    df_moti = melt_to_long(pd.read_excel(xls, 'improvementnecessary', header=1), 'Code', 'Moti_Index')
    
    df_emp_raw = pd.read_excel(xls, 'Employment by ILO', header=[0, 1])
    emp_list = []
    for col in df_emp_raw.columns:
        if str(col[0]).isdigit() and col[1] == 'Self-employed' and 2013 <= int(col[0]) <= 2023:
            temp = df_emp_raw[[('Unnamed: 0_level_0', 'Code'), col]].copy()
            temp.columns = ['Code', 'Self_Employment_Rate']
            temp['Year'] = int(col[0])
            emp_list.append(temp)
    df_emp_long = pd.concat(emp_list)
    df_emp_long['Self_Employment_Rate'] = df_emp_long['Self_Employment_Rate'].apply(clean_numeric)
    df_emp_long['Self_Employment_Rate'] = df_emp_long['Self_Employment_Rate'].apply(lambda x: x*100 if x < 1 and not pd.isna(x) else x)

    df = df_he.merge(df_unemp, on=['Code', 'Year'], how='outer') \
             .merge(df_emp_long, on=['Code', 'Year'], how='outer') \
             .merge(df_tea, on=['Code', 'Year'], how='outer') \
             .merge(df_moti, on=['Code', 'Year'], how='outer')

    df = df.sort_values(['Code', 'Year'])
    for col in ['HE_Rate', 'Unemployment_Rate', 'Self_Employment_Rate', 'TEA', 'Moti_Index']:
        df[col] = df.groupby('Code')[col].transform(lambda x: x.interpolate(limit_direction='both', limit=1))

    df_code = pd.read_excel(xls, 'OECD 38 國 ISO Code 對照表')[['ISO Code', '區域']].rename(columns={'ISO Code':'Code'})
    df = df.merge(df_code, on='Code', how='left')
    df['Is_East_Asia'] = df['Code'].apply(lambda x: 1 if x in ['TWN', 'JPN', 'KOR'] else 0)
    df['Group'] = df['Is_East_Asia'].map({1: 'East Asia', 0: 'Other OECD'})
    
    return df

# ==========================================
# 4. 圖表分析生成器 (Chart 1 - Chart 8)
# ==========================================
def generate_all_plots(df):
    log("開始產出 8 張核心研究圖表 (含 2023 四象限分析)...")
    
    # 圖 1 & 2: 2022 背景與四象限
    df22 = df[df['Year'] == 2022].dropna(subset=['Unemployment_Rate', 'Self_Employment_Rate'])
    v_avg22, h_avg22 = df22['Unemployment_Rate'].mean(), df22['Self_Employment_Rate'].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.regplot(x='HE_Rate', y='Self_Employment_Rate', data=df22, ax=ax1, scatter_kws={'alpha':0.4})
    ax1.set_title("Chart 1: 全球高教擴張與自雇率溢出關係 (2022)")
    
    ax2.axvline(v_avg22, color='grey', ls='--'); ax2.axhline(h_avg22, color='grey', ls='--')
    sns.scatterplot(x='Unemployment_Rate', y='Self_Employment_Rate', hue='Group', size='TEA', sizes=(50, 500), data=df22, ax=ax2, palette='coolwarm')
    for code in ['TWN', 'JPN', 'KOR']:
        p = df22[df22['Code'] == code]
        if not p.empty: ax2.text(p['Unemployment_Rate'].iloc[0]+0.15, p['Self_Employment_Rate'].iloc[0], f"{code}", weight='bold')
    ax2.set_title("Chart 2: 失業壓力 vs 自雇出口 (2022 四象限)")
    plt.savefig(f"{OUTPUT_DIR}/plot_01_02_2022_context.png")

    # 圖 3: 東亞趨勢
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df[df['Is_East_Asia']==1], x='Year', y='Self_Employment_Rate', hue='Code', marker='o', linewidth=2)
    plt.title("Chart 3: 東亞三國自雇率變遷軌跡 (2013-2023)")
    plt.savefig(f"{OUTPUT_DIR}/plot_03_ea_trends.png")

    # 圖 4: 台灣深度故事
    twn = df[df['Code'] == 'TWN'].sort_values('Year')
    fig, ax3 = plt.subplots(figsize=(10, 5))
    ax4 = ax3.twinx()
    ax3.plot(twn['Year'], twn['HE_Rate'], 'g-s', label='高教在學率', linewidth=2)
    ax4.plot(twn['Year'], twn['TEA'], 'r-^', label='TEA 指數', linewidth=2)
    plt.title("Chart 4: 台灣個案連動分析")
    ax3.legend(loc='upper left'); ax4.legend(loc='upper right')
    plt.savefig(f"{OUTPUT_DIR}/plot_04_taiwan.png")

    # 圖 5: 創業動機 (2018)
    plt.figure(figsize=(9, 6))
    df18 = df[df['Year'] == 2018].dropna(subset=['Moti_Index', 'TEA'])
    sns.scatterplot(x='Moti_Index', y='TEA', hue='Group', s=150, data=df18)
    plt.axvline(df18['Moti_Index'].mean(), color='gray', ls='--')
    plt.title("Chart 5: 創業性質：機會型 vs 生存型 (2018)")
    plt.savefig(f"{OUTPUT_DIR}/plot_05_motivation.png")

    # 圖 6: 調節效果 (2022)
    sns.lmplot(x='HE_Rate', y='Self_Employment_Rate', hue='Group', data=df22, aspect=1.4, palette='Set1')
    plt.title("Chart 6: 高教擴張之調節效果視覺化 (2022)")
    plt.savefig(f"{OUTPUT_DIR}/plot_06_interaction.png")

    # 圖 7: 疫情前後斜率對照
    df_comp = df[df['Year'].isin([2019, 2023])].copy()
    df_comp['Period'] = df_comp['Year'].map({2019: '疫情前 (2019)', 2023: '後疫情/AI (2023)'})
    sns.lmplot(x='HE_Rate', y='Self_Employment_Rate', hue='Group', col='Period', data=df_comp, aspect=1.2)
    plt.savefig(f"{OUTPUT_DIR}/plot_07_pandemic_comparison.png")

    # 圖 8: 新增的 2023 四象限分析
    df23 = df[df['Year'] == 2023].dropna(subset=['Unemployment_Rate', 'Self_Employment_Rate'])
    v_avg23, h_avg23 = df23['Unemployment_Rate'].mean(), df23['Self_Employment_Rate'].mean()
    plt.figure(figsize=(12, 8))
    plt.axvline(v_avg23, color='grey', ls='--', alpha=0.5); plt.axhline(h_avg23, color='grey', ls='--', alpha=0.5)
    sns.scatterplot(x='Unemployment_Rate', y='Self_Employment_Rate', hue='Group', size='TEA', sizes=(100, 1000), data=df23, palette='coolwarm', alpha=0.7)
    for code in ['TWN', 'JPN', 'KOR']:
        p = df23[df23['Code'] == code]
        if not p.empty: plt.text(p['Unemployment_Rate'].iloc[0]+0.1, p['Self_Employment_Rate'].iloc[0], f"{code} (TEA:{p['TEA'].iloc[0]:.1f})", weight='bold')
    plt.title("Chart 8: 2023 全球失業壓力與自雇出口象限圖", fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/plot_08_quadrant_2023.png")

# ==========================================
# 5. 主執行程序
# ==========================================
if __name__ == "__main__":
    try:
        final_df = build_thesis_database()
        generate_all_plots(final_df)
        final_df.to_csv(f"{OUTPUT_DIR}/final_dataset_2023.csv", index=False, encoding='utf-8-sig')
        log(f"🎉 全部生成完畢！請至 '{OUTPUT_DIR}' 資料夾查看。")
    except Exception as e:
        log(f"❌ 錯誤: {e}")
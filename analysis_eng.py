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
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 確保支援中文顯示
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
# 4. 圖表分析生成器 (All English Version)
# ==========================================
def generate_all_plots(df):
    log("開始產出 5 張核心假設驗證圖表 (全英文版本)...")
    
    # ---------------------------------------------------------
    # Chart 1: The Interaction Plot
    # ---------------------------------------------------------
    log("Drawing Chart 1...")
    df_h1_mean = df.groupby(['Code', 'Group'])[['HE_Rate', 'Self_Employment_Rate']].mean().reset_index()
    df_h1_mean = df_h1_mean.dropna(subset=['HE_Rate', 'Self_Employment_Rate'])

    hue_order = ['East Asia', 'Other OECD']
    marker_map = ['*', 'o']
    # 刪除了 line_map 的設定

    # 修改這裡：拿掉了 linestyles 參數
    g1 = sns.lmplot(
        x='HE_Rate', y='Self_Employment_Rate', 
        hue='Group', hue_order=hue_order, markers=marker_map, 
        data=df_h1_mean, aspect=1.5, palette='dark', scatter_kws={'s': 250, 'alpha': 0.7}
    )
    
    for _, row in df_h1_mean[df_h1_mean['Group'] == 'East Asia'].iterrows():
        plt.text(row['HE_Rate'] + 0.8, row['Self_Employment_Rate'], row['Code'], weight='bold', color='black', fontsize=10)

    g1.fig.suptitle("Institutional Moderation Effect (2013-2023 Avg.)", y=1.05, fontsize=14)
    g1.set_axis_labels("Higher Education Enrollment Rate (%)", "Self-employment Rate (%)")
    
    plt.savefig(f"{OUTPUT_DIR}/plot_01_各國交互作用(Higher Education-Self-employment).png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Chart 2: The Qualitative Nature
    # ---------------------------------------------------------
    log("Drawing Chart 2...")
    df_mean = df.groupby(['Code', 'Group', 'Is_East_Asia'])[['Moti_Index', 'TEA', 'Self_Employment_Rate']].mean().reset_index()
    df_mean = df_mean.dropna(subset=['Moti_Index', 'TEA', 'Self_Employment_Rate'])
    
    x_avg = df_mean['Moti_Index'].mean()
    y_avg = df_mean['TEA'].mean()

    plt.figure(figsize=(11, 8))
    plt.axvline(x_avg, color='grey', ls='--', alpha=0.6, linewidth=1.5)
    plt.axhline(y_avg, color='grey', ls='--', alpha=0.6, linewidth=1.5)
    
    sns.scatterplot(x='Moti_Index', y='TEA', size='Self_Employment_Rate', hue='Group', 
                    sizes=(80, 1200), alpha=0.75, palette='Set1', data=df_mean)
    
    for _, row in df_mean[df_mean['Is_East_Asia'] == 1].iterrows():
        plt.text(row['Moti_Index'] + 0.15, row['TEA'], row['Code'], weight='bold', color='black', fontsize=11)
        
    plt.title("Structural Overflow Bubble Chart (2013-2023 Avg.)", fontsize=15, pad=15)
    plt.xlabel(f"Motivation Index [Global Mean: {x_avg:.2f}]", fontsize=12)
    plt.ylabel(f"TEA Index [Global Mean: {y_avg:.2f}]", fontsize=12)

    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Legend (Bubble Size = Self-employment)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_02_TEA-Motives.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Chart 3: The Mechanism Quadrant
    # ---------------------------------------------------------
    log("Drawing Chart 3...")
    df23 = df[df['Year'] == 2023].dropna(subset=['Unemployment_Rate', 'Self_Employment_Rate'])
    v_avg23, h_avg23 = df23['Unemployment_Rate'].mean(), df23['Self_Employment_Rate'].mean()
    
    plt.figure(figsize=(12, 8))
    plt.axvline(v_avg23, color='grey', ls='--', alpha=0.5, linewidth=1.5)
    plt.axhline(h_avg23, color='grey', ls='--', alpha=0.5, linewidth=1.5)
    
    sns.scatterplot(x='Unemployment_Rate', y='Self_Employment_Rate', hue='Group', 
                    size='TEA', sizes=(100, 1200), data=df23, 
                    palette='Set1', alpha=0.85, hue_order=['East Asia', 'Other OECD'])
    
    for code in ['TWN', 'JPN', 'KOR']:
        p = df23[df23['Code'] == code]
        if not p.empty: 
            plt.text(p['Unemployment_Rate'].iloc[0] + 0.1, p['Self_Employment_Rate'].iloc[0], 
                     f"{code} (TEA:{p['TEA'].iloc[0]:.1f})", weight='bold', fontsize=11, color='black')
            
    plt.title("The Mechanism Quadrant (2023)", fontsize=16, pad=15)
    plt.xlabel(f"Unemployment Rate (%) [Mean: {v_avg23:.2f}]", fontsize=12)
    plt.ylabel(f"Self-employment Rate (%) [Mean: {h_avg23:.2f}]", fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Group & TEA Size")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_03_Unemployment-Self-employment with TEA.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Chart 4: The AI Shock
    # ---------------------------------------------------------
    log("Drawing Chart 4...")
    df_comp = df[df['Year'].isin([2019, 2023])].copy()
    df_comp['Period'] = df_comp['Year'].map({2019: '1. 2019 (Pre-Pandemic)', 2023: '2. 2023 (AI Shock)'})
    
    # 設定標記與顏色（延續圖一的雙重編碼：東亞星號、其他圓點）
    hue_order = ['East Asia', 'Other OECD']
    marker_map = ['*', 'o']
    g4 = sns.lmplot(
        x='HE_Rate', y='Self_Employment_Rate', hue='Group', col='Period', data=df_comp, 
        hue_order=hue_order, markers=marker_map, palette='Set1', aspect=1.2, 
        scatter_kws={'s': 200, 'alpha': 0.6}, facet_kws={'sharey': True, 'sharex': True}, ci=None
    )

    for period_name, ax in g4.axes_dict.items():
        subset = df_comp[(df_comp['Period'] == period_name) & (df_comp['Is_East_Asia'] == 1)]
        for _, row in subset.iterrows():
            ax.text(row['HE_Rate'] + 0.8, row['Self_Employment_Rate'], row['Code'], weight='bold', fontsize=9, color='black')

    g4.fig.suptitle("Slope Comparison - The AI Shock (2019 vs 2023)", y=1.08, fontsize=15)
    g4.set_axis_labels("Higher Education Enrollment Rate (%)", "Self-employment Rate (%)")
    plt.savefig(f"{OUTPUT_DIR}/plot_04_疫情前後對照.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Chart 5: The Career Reservoir
    # ---------------------------------------------------------
    log("Drawing Chart 5...")
    ea_codes = ['TWN', 'JPN', 'KOR']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, code in enumerate(ea_codes):
        country_data = df[df['Code'] == code].sort_values('Year')
        if country_data.empty: continue
            
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        p1 = ax1.axvspan(2019.8, 2022.2, color='#fef08a', alpha=0.5, label='Pandemic (2020-2022)')
        
        l1 = ax1.plot(country_data['Year'], country_data['HE_Rate'], 'g-s', label='HE Rate (%)', linewidth=2.5)[0]
        l2 = ax2.plot(country_data['Year'], country_data['TEA'], 'r-^', label='TEA Index', linewidth=2.5)[0]
        
        ax1.set_title(f"{code}", fontsize=14, weight='bold')
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Higher Education Enrollment Rate (%)", color='g')
        ax2.set_ylabel("TEA Index", color='r')
        
        ax1.set_xticks(range(2013, 2024, 2))
        
        if i == 0: 
            handles = [p1, l1, l2]
            labels = [h.get_label() for h in handles]
            ax1.legend(handles, labels, loc='upper left', framealpha=0.9)

    plt.suptitle("Temporal Deep-Dive of East Asia - Career Reservoir Effect", fontsize=16, y=1.08)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_05_東亞三國比較.png", bbox_inches='tight')
    plt.close()

# ==========================================
# 5. 主執行程序
# ==========================================
if __name__ == "__main__":
    try:
        final_df = build_thesis_database()
        generate_all_plots(final_df)
        final_df.to_csv(f"{OUTPUT_DIR}/final_dataset_eng_2023.csv", index=False, encoding='utf-8-sig')
        log(f"🎉 全部 5 張核心圖表生成完畢！請至 '{OUTPUT_DIR}' 資料夾查看。")
    except Exception as e:
        log(f"❌ 錯誤: {e}")
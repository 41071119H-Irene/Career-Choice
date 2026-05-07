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
# 4. 圖表分析生成器 (新 5 張核心圖表)
# ==========================================
def generate_all_plots(df):
    log("開始產出 5 張核心假設驗證圖表...")
    
   # ---------------------------------------------------------
    # 圖 1: 高教—自僱：制度調節效果圖 (The Interaction Plot)
    # 對應假設: H1 (Institutional Moderation)
    # ---------------------------------------------------------
    log("繪製 圖1: 制度調節效果圖 (使用跨年度平均)...")
    
    # 計算各國 2013-2023 的平均值，過濾短期波動雜訊
    df_h1_mean = df.groupby(['Code', 'Group'])[['HE_Rate', 'Self_Employment_Rate']].mean().reset_index()
    df_h1_mean = df_h1_mean.dropna(subset=['HE_Rate', 'Self_Employment_Rate'])

    # 繪製迴歸與散佈圖 (加大點的尺寸 s=80)
    g1 = sns.lmplot(x='HE_Rate', y='Self_Employment_Rate', hue='Group', data=df_h1_mean, 
                    aspect=1.5, palette='Set1', scatter_kws={'s': 80, 'alpha': 0.7})
    
    # 標示出東亞國家的點，讓視覺焦點更明確
    for _, row in df_h1_mean[df_h1_mean['Group'] == 'East Asia'].iterrows():
        plt.text(row['HE_Rate'] + 0.8, row['Self_Employment_Rate'], row['Code'], weight='bold', color='black')

    g1.fig.suptitle("圖1: 高教—自僱：制度調節效果圖 (2013-2023 平均)\n[對應假設: H1 Institutional Moderation]", y=1.05, fontsize=14)
    g1.set_axis_labels("高等教育在學率 (HE Rate, %)", "自僱率 (Self-employment Rate, %)")
    plt.savefig(f"{OUTPUT_DIR}/plot_01_interaction_effect.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 圖 2: TEA—動機：人力資本錯置氣泡圖 (The Qualitative Nature - 四象限版)
    # 對應假設: H2 (Structural Overflow)
    # ---------------------------------------------------------
    log("繪製 圖2: 人力資本錯置氣泡圖 (四象限版)...")
    # 計算 2013-2023 的各國平均值
    # 這裡加上 'Is_East_Asia' 確保後續迴圈讀取不會報錯
    df_mean = df.groupby(['Code', 'Group', 'Is_East_Asia'])[['Moti_Index', 'TEA', 'Self_Employment_Rate']].mean().reset_index()
    df_mean = df_mean.dropna(subset=['Moti_Index', 'TEA', 'Self_Employment_Rate'])
    
    # 計算全體國家的 Moti_Index (X軸) 與 TEA (Y軸) 平均值
    x_avg = df_mean['Moti_Index'].mean()
    y_avg = df_mean['TEA'].mean()

    plt.figure(figsize=(11, 8)) # 稍微放大版面，讓四象限看起來更大氣
    
    # 繪製象限基準線 (十字虛線)
    plt.axvline(x_avg, color='grey', ls='--', alpha=0.6, linewidth=1.5)
    plt.axhline(y_avg, color='grey', ls='--', alpha=0.6, linewidth=1.5)
    
    # 繪製氣泡圖 (微調氣泡大小的上限，讓視覺張力更好)
    sns.scatterplot(x='Moti_Index', y='TEA', size='Self_Employment_Rate', hue='Group', 
                    sizes=(80, 1200), alpha=0.75, palette='Set1', data=df_mean)
    
    # 標示東亞國家的點 (TWN, JPN, KOR)
    for _, row in df_mean[df_mean['Is_East_Asia'] == 1].iterrows():
        plt.text(row['Moti_Index'] + 0.15, row['TEA'], row['Code'], 
                 weight='bold', color='black', fontsize=11)
        
    plt.title("圖2: TEA—動機：人力資本錯置四象限氣泡圖 (2013-2023 平均)\n[對應假設: H2 Structural Overflow]", fontsize=15, pad=15)
    plt.xlabel(f"Moti Index (創業動機指數)  [平均基準線: {x_avg:.2f}]", fontsize=12)
    plt.ylabel(f"TEA (早期創業活動指數)  [平均基準線: {y_avg:.2f}]", fontsize=12)

    
    # 將圖例移到圖外
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="指標說明")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_02_structural_overflow_quadrant.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 圖 3: 2023 全球象限圖：勞動力洩壓閥機制 (The Mechanism Quadrant)
    # 對應假設: H2 (The Pressure Relief Valve)
    # ---------------------------------------------------------
    log("繪製 圖3: 2023 全球象限圖 (統一高對比鮮豔色系)...")
    df23 = df[df['Year'] == 2023].dropna(subset=['Unemployment_Rate', 'Self_Employment_Rate'])
    v_avg23, h_avg23 = df23['Unemployment_Rate'].mean(), df23['Self_Employment_Rate'].mean()
    
    plt.figure(figsize=(12, 8))
    
    # 繪製十字基準線
    plt.axvline(v_avg23, color='grey', ls='--', alpha=0.5, linewidth=1.5)
    plt.axhline(h_avg23, color='grey', ls='--', alpha=0.5, linewidth=1.5)
    
    # 統一使用 Set1 調色盤，確保東亞組的顏色與前兩張圖完全相同
    # alpha 調高至 0.85 讓氣泡顏色更實心、更鮮豔
    sns.scatterplot(x='Unemployment_Rate', y='Self_Employment_Rate', hue='Group', 
                    size='TEA', sizes=(100, 1200), data=df23, 
                    palette='Set1', alpha=0.85, hue_order=['East Asia', 'Other OECD'])
    
    # 標示東亞三國 (字體加黑、加大，避免被鮮豔的氣泡吃掉)
    for code in ['TWN', 'JPN', 'KOR']:
        p = df23[df23['Code'] == code]
        if not p.empty: 
            plt.text(p['Unemployment_Rate'].iloc[0] + 0.1, p['Self_Employment_Rate'].iloc[0], 
                     f"{code} (TEA:{p['TEA'].iloc[0]:.1f})", weight='bold', fontsize=11, color='black')
            
    plt.title("圖3: 2023 全球象限圖：勞動力洩壓閥機制\n[對應假設: H2 The Pressure Relief Valve]", fontsize=16, pad=15)
    plt.xlabel(f"失業率 (Unemployment Rate, %)  [平均基準線: {v_avg23:.2f}]", fontsize=12)
    plt.ylabel(f"自僱率 (Self-employment Rate, %)  [平均基準線: {h_avg23:.2f}]", fontsize=12)
    
    # 將圖例移到圖外，避免遮擋氣泡，並加上標題
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="國家群組與 TEA 氣泡大小")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_03_mechanism_quadrant_2023.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 圖 4: 2019 vs 2023 斜率對照圖：技術變革下的職涯衝擊 (The AI Shock)
    # 對應假設: H3 (Intensification Effect)
    # ---------------------------------------------------------
    log("繪製 圖 4: 2019 vs 2023 斜率對照標註版...")
    
    # 準備對比數據
    df_comp = df[df['Year'].isin([2019, 2023])].copy()
    # 建立一個明確的排序標籤，確保 2019 在左，2023 在右
    df_comp['Period'] = df_comp['Year'].map({2019: '1. 2019 (Pre-Pandemic)', 2023: '2. 2023 (AI Shock)'})
    
    # 設定標記與顏色（延續圖一的雙重編碼：東亞星號、其他圓點）
    hue_order = ['East Asia', 'Other OECD']
    marker_map = ['*', 'o']
    
    g4 = sns.lmplot(
        x='HE_Rate', y='Self_Employment_Rate', 
        hue='Group', col='Period', 
        data=df_comp, 
        hue_order=hue_order,
        markers=marker_map,
        palette='Set1', 
        aspect=1.2, 
        scatter_kws={'s': 200, 'alpha': 0.6},
        facet_kws={'sharey': True, 'sharex': True} # 確保 X, Y 軸刻度一致，方便對比位移
    )

    # --- 關鍵步驟：在分面子圖中標註台日韓 ---
    # 遍歷每一個子圖 (ax) 與其對應的期別名稱 (title)
    for period_name, ax in g4.axes_dict.items():
        # 篩選該子圖對應年份的東亞資料
        subset = df_comp[(df_comp['Period'] == period_name) & (df_comp['Is_East_Asia'] == 1)]
        
        for _, row in subset.iterrows():
            # 加上國家代碼標籤
            # row['HE_Rate']+0.8 是為了把文字稍微往右移，避免擋到點
            ax.text(
                row['HE_Rate'] + 0.8, 
                row['Self_Employment_Rate'], 
                row['Code'], 
                weight='bold', 
                fontsize=9, 
                color='black'
            )

    g4.fig.suptitle("圖4: 2019 vs 2023 斜率對照圖：技術變革下的職涯衝擊\n[標註東亞三國位移狀況]", y=1.08, fontsize=15)
    g4.set_axis_labels("高等教育在學率 (HE Rate, %)", "自僱率 (Self-employment Rate, %)")
    
    plt.savefig(f"{OUTPUT_DIR}/plot_04_ai_shock_comparison_labeled.png", bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 圖 5: 東亞(台灣、日本、韓國)時序深度分析：職涯避風港效應
    # 對應理論: Social Buffer / Reservoir Hypothesis
    # ---------------------------------------------------------
    log("繪製 圖5: 東亞三國時序深度分析 (加入黃底疫情圖例)...")
    ea_codes = ['TWN', 'JPN', 'KOR']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, code in enumerate(ea_codes):
        country_data = df[df['Code'] == code].sort_values('Year')
        if country_data.empty: continue
            
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # 1. 繪製黃色疫情區塊，並存成變數 p1
        p1 = ax1.axvspan(2019.8, 2022.2, color='#fef08a', alpha=0.5, label='Pandemic (2020-2022)')
        
        # 2. 繪製雙 Y 軸線圖 (注意：plot 會回傳一個 list，所以我們加上 [0] 取出線條本體)
        l1 = ax1.plot(country_data['Year'], country_data['HE_Rate'], 'g-s', label='HE Rate (高等教育)', linewidth=2.5)[0]
        l2 = ax2.plot(country_data['Year'], country_data['TEA'], 'r-^', label='TEA (創業活躍度)', linewidth=2.5)[0]
        
        ax1.set_title(f"{code} 時序避風港分析", fontsize=14, weight='bold')
        ax1.set_xlabel("Year")
        ax1.set_ylabel("高等教育在學率 (%)", color='g')
        ax2.set_ylabel("創業活躍度 TEA", color='r')
        
        # 設定 X 軸刻度避免小數點年份
        ax1.set_xticks(range(2013, 2024, 2))
        
        # 3. 合併所有圖例 (只在第一張圖 TWN 顯示，避免畫面太雜)
        if i == 0: 
            # 將黃底、綠線、紅線的物件與名稱打包
            handles = [p1, l1, l2]
            labels = [h.get_label() for h in handles]
            
            # 放到左上角，並稍微加上一點背景白底避免與線條重疊看不清
            ax1.legend(handles, labels, loc='upper left', framealpha=0.9)

    plt.suptitle("圖5: 東亞時序深度分析：職涯避風港效應\n[對應理論: Social Buffer / Reservoir Hypothesis]", fontsize=16, y=1.08)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_05_career_reservoir_timeline.png", bbox_inches='tight')
    plt.close()

# ==========================================
# 5. 主執行程序
# ==========================================
if __name__ == "__main__":
    try:
        final_df = build_thesis_database()
        generate_all_plots(final_df)
        final_df.to_csv(f"{OUTPUT_DIR}/final_dataset_2023.csv", index=False, encoding='utf-8-sig')
        log(f"🎉 全部 5 張核心圖表生成完畢！請至 '{OUTPUT_DIR}' 資料夾查看。")
    except Exception as e:
        log(f"❌ 錯誤: {e}")
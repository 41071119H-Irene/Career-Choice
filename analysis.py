import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. 環境與字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

def clean_val(x):
    """清理數值：處理百分比、逗號與空值"""
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace('%', '').replace(',', '').strip()
        return float(x) if x != '' else np.nan
    return float(x)

def melt_data(df, id_col, value_name):
    """將寬表格轉為長表格，並確保年份為整數"""
    # 找出所有看起來像年份的欄位 (2011-2025)
    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    df_long = df.melt(id_vars=[id_col], value_vars=year_cols, var_name='Year', value_name=value_name)
    df_long['Year'] = df_long['Year'].astype(int)
    # 只取 2013-2023
    df_long = df_long[(df_long['Year'] >= 2013) & (df_long['Year'] <= 2023)]
    df_long[value_name] = df_long[value_name].apply(clean_val)
    return df_long

def load_and_process():
    print("正在讀取分頁 CSV 檔案...")
    
    # A. 根據你上傳的實際檔名讀取
    df_he = pd.read_csv('data.xlsx - School enrollment, tertiary % g.csv')
    df_unemp = pd.read_csv('data.xlsx - Unemployment, total (% of total.csv')
    df_tea = pd.read_csv('data.xlsx - Percentage of 18-64 population .csv', skiprows=1)
    df_moti = pd.read_csv('data.xlsx - improvementnecessary.csv', skiprows=1)
    df_code = pd.read_csv('data.xlsx - OECD 38 國 ISO Code 對照表.csv')
    
    # B. 處理最棘手的 Employment by ILO (雙層標題)
    # 直接用第 0, 1 列手動解析
    df_emp_raw = pd.read_csv('data.xlsx - Employment by ILO.csv', header=None)
    years_row = df_emp_raw.iloc[0].ffill() 
    types_row = df_emp_raw.iloc[1]
    
    emp_records = []
    for col in range(1, len(df_emp_raw.columns)):
        y = str(years_row[col])
        t = str(types_row[col])
        if y.isdigit() and 'Self-employed' in t:
            temp = df_emp_raw[[0, col]].iloc[2:].copy()
            temp.columns = ['Code', 'Self_Employment_Rate']
            temp['Year'] = int(y)
            emp_records.append(temp)
    df_emp_long = pd.concat(emp_records)
    df_emp_long['Self_Employment_Rate'] = df_emp_long['Self_Employment_Rate'].apply(clean_val)
    # 如果是小數 (如 0.14) 轉為百分比 (14)
    df_emp_long['Self_Employment_Rate'] = df_emp_long['Self_Employment_Rate'].apply(lambda x: x*100 if x < 1 else x)

    # C. 轉換其餘資料為長表格
    he_long = melt_data(df_he, 'Country Code', 'HE_Rate').rename(columns={'Country Code': 'Code'})
    unemp_long = melt_data(df_unemp, 'Country Code', 'Unemployment_Rate').rename(columns={'Country Code': 'Code'})
    tea_long = melt_data(df_tea, 'Code', 'TEA')
    moti_long = melt_data(df_moti, 'Code', 'Moti_Index')

    # D. 合併成 Panel Data (2013-2023)
    final_df = he_long.merge(unemp_long, on=['Code', 'Year'], how='outer') \
                      .merge(df_emp_long, on=['Code', 'Year'], how='outer') \
                      .merge(tea_long, on=['Code', 'Year'], how='outer') \
                      .merge(moti_long, on=['Code', 'Year'], how='outer')
    
    # 加入區域標籤
    df_code_clean = df_code[['ISO Code', '區域']].rename(columns={'ISO Code': 'Code', '區域': 'Region'})
    final_df = final_df.merge(df_code_clean, on='Code', how='left')
    
    return final_df

def plot_results(df):
    # --- 圖 1: 台灣 2013-2023 故事線 ---
    twn = df[df['Code'] == 'TWN'].sort_values('Year')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(twn['Year'], twn['HE_Rate'], 'g-s', label='高教在學率', linewidth=2)
    ax1.plot(twn['Year'], twn['Unemployment_Rate'], 'b--o', label='失業率')
    ax2.plot(twn['Year'], twn['TEA'], 'r-^', label='TEA 創業活躍度', linewidth=2)
    
    ax1.set_xlabel('年份'); ax1.set_ylabel('百分比 (%)'); ax2.set_ylabel('TEA Index')
    plt.title('台灣個案分析 (2013-2023)：高教擴張與創業活動趨勢')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.show()

    # --- 圖 2: 全球高教 vs 自雇散佈圖 (2022) ---
    df_2022 = df[df['Year'] == 2022].dropna(subset=['HE_Rate', 'Self_Employment_Rate'])
    plt.figure(figsize=(10, 6))
    sns.regplot(x='HE_Rate', y='Self_Employment_Rate', data=df_2022)
    # 標註東亞三國
    for code in ['TWN', 'JPN', 'KOR']:
        row = df_2022[df_2022['Code'] == code]
        if not row.empty:
            plt.text(row['HE_Rate'].iloc[0], row['Self_Employment_Rate'].iloc[0], code, color='red', weight='bold')
    plt.title('高等教育擴張對自雇率的影響 (2022 全球截面分析)')
    plt.show()

if __name__ == "__main__":
    try:
        full_data = load_and_process()
        plot_results(full_data)
        print("分析成功！已生成 2013-2023 的演變圖表。")
    except Exception as e:
        print(f"錯誤：{e}")
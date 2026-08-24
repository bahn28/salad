# %%
import pandas as pd
# 이것은 작동 csv file, 뒤에 ?raw=true 이걸 붙이는 군.
url1 = 'https://github.com/bahn28/potato/blob/main/cs_nns_gndr_hgt.csv?raw=true'
# 아래처럼 raw로 주소 지정 가능.
# url1raw = 'https://raw.githubusercontent.com/bahn28/potato/refs/heads/main/cs_nns_gndr_hgt.csv'
df_dat = pd.read_csv(url1) 
df_dat.head()

# url2 = 'https://github.com/bahn28/potato/blob/main/cs_nns_gndr_hgt.xlsx?raw=true'
# url3 = 'https://github.com/bahn28/potato/blob/main/cs_nns_gndr_wgt.xlsx?raw=true'
# url4 = 'https://github.com/bahn28/potato/blob/main/cs_nns_hgt_wgt.xlsx?raw=true'
# url5 = 'https://github.com/bahn28/potato/blob/main/cs_nns_wt_ht_gndr_age_full.xlsx?raw=true'
# url6 = 'https://github.com/bahn28/potato/blob/main/cs_prof_sal_exp_gndr.xlsx?raw=true'
# url7 = 'https://github.com/bahn28/potato/blob/main/ts_gdp_1971_2025.xlsx?raw=true'
# url8 = 'https://github.com/bahn28/potato/blob/main/ts_stock_date_price.xlsx?raw=true'
# url9 = 'https://github.com/bahn28/potato/blob/main/ts_sun_sel_dg.xlsx?raw=true'

# df_dat3 = pd.read_excel(url3) 
# df_dat3.head()


# %%
# 사전트 사이트 
# https://python.quantecon.org/intro.html
# topic 134 Pandas for Panel Data
# Display 6 columns for viewing purposes
import pandas as pd

url1 = 'https://github.com/QuantEcon/data-lectures/raw/main/lectures/realwage.csv'
# Display 6 columns for viewing purposes
pd.set_option('display.max_columns', 6)
realwage = pd.read_csv(url1)

# %%
print(realwage.head())  # Show first 5 rows




# %%

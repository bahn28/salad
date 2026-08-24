#%%
# _tsa_00.py
# time series stuff, basic 
# 그리기
# 시계열, 추세, 계절성 조정, 이동평균, 이런 것들. 
# 자기공분산, 자기상관, 단위근 
# ARIMA 중 AR(p)
# 기타... 단순한 내용만 간략하게. 


import os
import numpy as np                          # numpy 라이브러리 전체. 
import pandas as pd                         # pandas 라이브러리 전체, 시계열(기본) 여기 있네. 
import matplotlib.pyplot as plt
from scipy import stats, optimize, linalg   # scipy 라이브러리 하부 모듈.
from statsmodels.tsa import stattools       # time series analysis 
from statsmodels.tsa.arima.model import ARIMA # 이거 가능하다 이거지. 이거 대문자네. 꼭. 
from statsmodels.tsa.ar_model import AutoReg

#%%
# def generate_figure(x, y, title, xlabel, ylabel, output_path=None):
#     """x, y 데이터를 그래프로 그리며 선택적으로 파일로 저장합니다."""
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, y, color='blue', linewidth=2)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()

#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, dpi=300)
#         print(f"생성된 그래프를 이미지로 저장했습니다: {output_path}")

#     plt.show()

def load_data():
    file = os.path.join(input_dir, input_file)
    dat = pd.read_excel(file, sheet_name=input_sheet)
    return dat

#%%
def main():

# 1. 데이터 준비. 읽기. 생성. 
#   dat을 한 번에 읽었다, 3개 변수로 지정. dat은 data frame 형식.
    ddf = pd.read_excel(os.path.join(input_dir, input_file), sheet_name=input_sheet) 
    yy, gdp = [ddf[col] for col in ['yy', 'gdp']] 


    #ddf = load_data() # 별도 정의된 함수로 읽어 들이기

# 연습 데이터 생생 
    np.random.seed(seed)
#    nobs = 100
#    rho = 0.9

    eps = np.random.normal(scale=np.sqrt(1-rho**2), size=nobs)

    y = np.zeros(nobs)
    x = np.zeros(nobs)

    for t in range(1, nobs): 
        x[t] = t
        y[t] = rho*y[t-1] + eps[t]

    #print(y, " \n", len(y))

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='blue', linewidth=2)
    #plt.show()

    rhatmat = np.corrcoef(y[:-1], y[1:])   # :-1 ==> 마지막 제외, 1: 첫 제외
    #print(rhatmat)
    #np.corrcoef(y[:-k], y[k:])   # 마지막 k, 첫 k 제외. 

# autoregression
#    import pandas as pd
    # ddf = dat in dataformat, dnd = dat in ndarrry 버릇처럼 정해? 이런 게 해빗.
    # df = pd.read_csv("data.csv")  이름을 규칙적으롭 부르는 것도... 스타일이지...

    y = ddf["gdp"]
    # AR(p) model
    model = AutoReg( 
        y, 
        lags=2,
        trend="c"
        )
    result = model.fit()

    print( "AR(p) estimates ") # results 
    print(result.summary() ) #.params)

    print(" prediction over 4 periods")
    future = result.predict(
        start=len(y),
        end=len(y)+3
        )
    print(future)

#%%
if __name__ == "__main__":
    # 실행 전 설정
    output_fig_dir = 'figures'
    output_fig_file = 'sta_90001_tsa_00.png'
    output_dir = 'o_files'
    input_dir = 'in_files' 
    input_file = 'ts_gdp_1971_2025.xlsx' # 변수명: 체크 
    input_sheet = 'Sheet1'                       # yy, gdp  
    # input_file = 'ts_sun_sel_dg.xlsx' # 변수명: 체크 , 월 일조시간 
    # input_sheet = 'Sheet1'                       # yymm, sel, dgu
    # input_file = 'ts_stock_date_price.xlsx' # 변수명: 체크 , S전자 주가, 종가.
    # input_sheet = 'Sheet1'                       # yydd, price 
    # input_file = 'cs_prof_sal_exp_gndr.xlsx' # 변수명: 체크 
    # input_sheet = 'ISEG'                       # Individual, Salary, Exper, Gender 
#    input_file = 'cs_nns_hgt_wgt.xlsx' # 변수명: 체크 
#    input_sheet = 'hwght'                       # x->height, y->weight 
#    input_file = 'cs_nns_i_gndr_age_wt_ht_full.xlsx' # 변수명: 체크 
#    input_sheet = 'igawht'                     # i, gender, age, wt, ht  
    seed = 12348215 
    nobs = 1001 
    rho = 0.8

    main()

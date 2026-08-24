#%%
# _rd_01.py 
# 상관관계, 상관계수, 공분산, 산포도. 그룹, 히스토그램 등 회귀분석 전 설명.
# raw data - x: 키, y: 체중, n = 3111 
# mean function in regression. 작업예정... 

import os
import numpy as np                          # numpy 라이브러리 전체. 
import pandas as pd                         # pandas 라이브러리 전체, 시계열(기본) 여기 있네. 
import scipy as sci 
from scipy import stats, optimize, linalg   # scipy 라이브러리 하부 모듈. 필요한 통계 모듈만.
import statsmodels.api as sm                # 방대한 모델이라 개발자들이 많이 쓰는 모델 묶음.
from statsmodels.tsa import stattools       # time series analysis 
from statsmodels.tsa.arima.model import ARIMA # 이거 가능하다 이거지. 이거 대문자네. 꼭. 소문자 못읽어.
import matplotlib.pyplot as plt

#%%
def save_chart(fname, outdir, fig=None, dpi=300, bbox_inches='tight', transparent=False):
    if fig is None:
        fig = plt.gcf()
    os.makedirs(outdir, exist_ok=True)
    name, ext = os.path.splitext(fname)  # fname(파일)을 이름과 익스텐션으로 분리.
    ext = ext or '.png'
    path = os.path.join(outdir, name + ext)  # outdir에 fname을 짤라서 그림 이름으로 지정하네? 
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    return path


def main():
# 1. 데이터 준비. 읽기. 생성. 
#   데이터 읽고 처리 단계.
#   dat을 한 번에 읽었다, 3개 변수로 지정. dat은 data frame 형식.
    dat = pd.read_excel(os.path.join(input_dir, input_file), sheet_name=input_sheet) 
    n_obs = len(dat) 
    hgt = dat['x'].to_numpy()
    wgt = dat['y'].to_numpy()
#    gndr = (gndrlit == 'Male').astype(int)   # astype(int)으로 NumPy 배열 전환, 1/0  


# 2. 자료 검토, 요약, 그래프 생성.
# 2a. 자료 검토, 기초 통계량
    summ = dat.agg(
            ["count", "mean", "std", "min", "median", "max"]
            ) # 일부 변수, 일부 통계량 
    print(summ)

# 2b. 상관계수, 공분산, 전체 
    covmat = np.cov(wgt, hgt)
    cov = covmat[0, 1]
    rhomat = np.corrcoef(wgt, hgt)
    rho = rhomat[0,1]
#    print( "covmat\n", covmat, "\nrhomat\n", rhomat)
    print(f"covmat\n{covmat}\n\nrhomat\n{rhomat}") 

# 2c. 자료 검토, 시각화 확인.
# 산포도 ( x, y )  
    plt.figsize=(10, 10)
    plt.scatter(hgt, wgt, s=20, facecolor='none', edgecolor='black')
    plt.title(" Weight against height ", fontsize=11)
    plt.xlabel(" height ")
    plt.ylabel(" weight ")
    plt.show()
#        ax.scatter(
#            x, y, s=25, 
#            marker='o',
#            facecolor='none', edgecolor='black', linewidths=1.5
#            #color='black', 요건 점으로 할 때.
#            #alpha=0.7
#            )
#        ax.set_title(f"r = {r:.3f}", fontsize=11)
#        ax.set_xlim(-3.5, 3.5)
#        ax.set_ylim(-3.5, 3.5)
#        ax.set_aspect('equal')
#        ax.set_xticks([])
#        ax.set_yticks([])
#%%
# 3. 아웃풋 저장, 필요시 
#  저장 파일 이름 지정
#    saved_path = save_chart(
#        fname = output_fig_file,  # 메인() 밖에서 지정 'sta_09001_corr_01.png'
#        outdir = output_fig_dir, # 메인() 밖에서 지정 'figures', 
#        fig=fig, 
#        dpi=300)


# 4. 코드 체크
#  변수 특성, 배열 형식, 계산 완료 등.
#    print('Saved figure to', saved_path)
    print("Computing OK")

# 5. 메인() 일괄 실행. 

if __name__ == "__main__":
    # 실행 전 설정
    # 미리 준비하면서 정보를 파악해 두어야 함.
    # 아웃풋 폴더
    # 데이터 폴더, 데이터 파일 이름, 데이터 시트 이름,
    # 데이터의 그룹, 층, 범주 이름 등
    # 난수생성관련 조건 등 
    output_fig_dir = 'figures'
    output_fig_file = '0test.png'
    output_dir = 'o_files'
    input_dir = 'in_files' 
    input_file = 'cs_nns_hgt_wgt.xlsx' # 변수명: 체크 
    input_sheet = 'hwght'                       # x->height, y->weight 
#    input_file = 'cs_nns_i_gndr_age_wt_ht_full.xlsx' # 변수명: 체크 
#    input_sheet = 'igawht'                     # i, gender, age, wt, ht  
    seed = 12348215 


    main()

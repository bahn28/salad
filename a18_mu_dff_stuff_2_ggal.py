#%%
# # _rd_00.py 
# 비율 차이 추론. 평균차 분산, 표준오차, 신뢰구간, 가설검정  
#   등분산 전제, 이분산 전제. 오류 정정(자유도 관련 계산)

import os
import numpy as np                          # numpy 라이브러리 전체. 
import pandas as pd                         # pandas 라이브러리 전체, 시계열(기본) 여기 있네. 
import scipy as sci 
from scipy import stats, optimize, linalg   # scipy 라이브러리 하부 모듈. 필요한 통계 모듈만.
import statsmodels.api as sm                # 방대한 모델이라 개발자들이 많이 쓰는 모델 묶음.
from statsmodels.tsa import stattools       # time series analysis 
from statsmodels.tsa.arima.model import ARIMA # 이거 가능하다 이거지. 이거 대문자네. 꼭. 소문자 못읽어.
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.weightstats import DescrStatsW, ztest
from scipy.stats import ttest_1samp, ttest_ind, f

#%%
def main():
# 1. 데이터 준비. 읽기. 생성. 
#   dat을 한 번에 읽었다, 3개 변수로 지정. dat은 data frame 형식.
    dat_all = pd.read_excel(os.path.join(input_dir, input_file), sheet_name=input_sheet) 
    n_obs = len(dat_all)

# 1.1. 전체 자료 가운데 일부만 선택, 단순추출, 층화추출. 이번에는 층화추출
#      연습용.  
    n_frac = 0.02   # 2%, 몇 개만 뽑아서 보자는 말. 관측치 2만개 넘음. 
    n_sel = int(n_obs*n_frac)  # 단순추출이라면 사용.
#    dat = dat_all.sample(n_sel, replace=False, random_state=42)
    dat = (dat_all
           .groupby('gender', group_keys=False)
           .sample(frac=n_frac, random_state=42)
           )

#    dat.info()                     # 변수 3개: i, gender, ht. 
#   id = dat['i'].to_numpy()        # 그냥 번호. 별도 불요.
    gn = dat['gender'].to_numpy()   # 이렇게 하면 배열로 전환. dataframe -> NumPy 배열
    hgt = dat['ht'].to_numpy()

# 2. groups: by gndr (gender, 1 - 0), 
    gndr = (gn == 1).astype(int)    # gn = 1, 2; gndr = 1, 0. 1/0으로 전환. 
    dat['gndr'] = gndr               #   이것은 위에 1/0 자료를 추가. 성별임.

    # 그룹을 평균키 기준으로 구분하는 아이디어. 연습.
    hgt_ref = np.mean(hgt)          # 그룹 구분 참고값. 평균 이상, 이하.
    hgt_grp = np.where(hgt > hgt_ref, "Tall", "Short")   # 기준 키 대비 Tall, Short.
    dat['ht_g'] = hgt_grp            # data frame에 변수 추가된 것임. 
    mf_tall = dat[dat['ht_g'] == 'Tall']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  
    mf_shrt = dat[dat['ht_g'] == 'Short']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  

    # dat.info()  # 5개 변수 
    n_tall = (dat['ht_g'] == 'Tall').sum()
    m_tall = ((dat['ht_g'] == 'Tall') & (dat['gndr'] == 1)).sum() 
    n_shrt = len(dat) - n_tall 
    m_shrt = ((dat['ht_g'] == 'Short') & (dat['gndr'] == 1)).sum() 
    m_pct_tall = m_tall / n_tall 
    m_pct_shrt = m_shrt / n_shrt 
    print(" m1  m2  n1  n2 ")
    print(m_tall, m_shrt, n_tall, n_shrt) 

    diff_p = m_pct_tall - m_pct_shrt 
    pool_p = (m_tall + m_shrt)/(n_tall + n_shrt) 
    var_diff = pool_p * ( 1- pool_p) *( 1 / n_tall +  1 / n_shrt )
    se_diff = var_diff**0.5 
    diff_p_0 = 0 
    t0 = ( diff_p - diff_p_0 ) / se_diff  
    pval = 2 * (1 - stats.norm.cdf(abs(t0)) )
    print("t statisticse:", t0 )
    print("p-value:",  pval)


    # m f ratio out of tall and short 
    from statsmodels.stats.proportion import proportions_ztest
    count = [m_tall, m_shrt] 
    nobs = [n_tall, n_shrt]

    z_stat, p_value = proportions_ztest(count, nobs)
    print("z-statistic:", z_stat)
    print("p-value:", p_value)

#%%

#     n_all = len(dat) 
#     n_m = len(hgt_m)
#     n_f = len(hgt_f)

# # 3. 등분산 검정, 정의된 함수 이용. 위 셀.
#     phi_hat, pval, ci = variance_ratio_test( hgt_m, hgt_f, alpha) 
#     print(" 등분산 검정. F ratio, p value :", phi_hat, pval)
#     print(" 등분산 " if pval > alpha else " 이분산 ")

# # 분포 임계치, 양방향, 우측값. 적당한 위치 모색.
# #    alpha = 0.05                          # significance level, two side.
# #    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
# #    tcv_r = stats.t.ppf(1 - alpha / 2, df= n_tall)

# #%%
# # 4. 평균, 그룹별 차이 ( m - f ), 표준오차. 
# # 추정치: 평균, 분산, 표준오차, overall
# # overall, 이 계산은 불필요함. 
#     muhat = hgt.mean()
#     var_hgt = hgt.var(ddof=1)  #/  n_all # 이게 통합 분산은 아니지. 

# # 그룹별( m - f) 표본평균 및 분산
#     muhat_m = hgt_m.mean()
#     muhat_f = hgt_f.mean()
#     var_hgt_m = hgt_m.var(ddof=1) # /  n_m 
#     var_hgt_f = hgt_f.var(ddof=1) #/  n_f 
# # 평균차이
#     mu_diff = muhat_m - muhat_f 


# # 그룹 평균차의 분산, 표준오차 (등분산 vs. 이분산)
# #   등분산 가정.
#     df_pool = n_m + n_f - 2                                    # 등분산 자유도 
#     sse_pool = (n_m - 1) * var_hgt_m + (n_f - 1) * var_hgt_f   # 등분산 가정 시 적용 가능 
#     var_pool = sse_pool / df_pool                              # 공통분산 
#     var_eq =  var_pool * ( 1 / n_m + 1 / n_f )     # 평균차의 분산(등분산인 경우). 
#     se_eq = np.sqrt( var_eq )                      # 평균차의 표준오차

# #   이분산 가정.
#     var_un =  var_hgt_m / n_m + var_hgt_f / n_f    # 평균차이 분산, 이분산(일반적) 
#     se_un = np.sqrt( var_un )                      #         표분오차
#     df_hetr =  var_un**2 / ( 
#         (var_hgt_m/n_m)**2/(n_m-1 )
#         + (var_hgt_f/n_f)**2/(n_f-1) )             #   이분산 자유도,  Welch–Satter 방법 

# # 평균차이, 표준오차 
#     print("평균 차이, 등분산: (표준오차, 자유도), 이분산: (표준오차, 자유도)")
#     print( mu_diff, se_eq, df_pool, se_un, df_hetr) 

# # 신뢰구간 찾기, right, left, 등분산, 이분산. 
# # 임계치 
#     tcv_r_eq = stats.t.ppf( 1- alpha/2, df_pool )  # critical value on the right, two side
#     tcv_r_un = stats.t.ppf( 1- alpha/2, df_hetr )  # critical value on the right, two side
# #    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
# # 신뢰구간(등분산)
#     ci_r = mu_diff + tcv_r_eq * se_eq 
#     ci_l = mu_diff - tcv_r_eq * se_eq   

#     print( "등분산 가정:")
#     print( "   mean diff,    se of mu_diff,     confidence interval ")
#     print( mu_diff, se_eq, ci_l, ci_r ) 

# # 신뢰구간(이분산) --- 임계치와 표준오차 조정/변경/선택, 자유도...  
#     ci_r = mu_diff + tcv_r_un * se_un 
#     ci_l = mu_diff - tcv_r_un * se_un   

#     print( "이분산 가정:")
#     print( "HETR  mean diff,    se of mu_diff,     confidence interval ")
#     print( mu_diff, se_un, ci_l, ci_r ) 

# #%%
# # 가설검정. 전체, 톨 
# # 귀무가설 H0: mu = mu0
# # 검정통계치, t_0, pvalue 
#     print("H0: 두 그룹 평균이 동일하다")
#     mu_diff_zero = 0

#     t_0eq = np.abs( ( mu_diff - mu_diff_zero ) / se_eq  )       # 등분산 
#     t_0un = np.abs( ( mu_diff - mu_diff_zero ) / se_un  )       # 이분산

#     yn_h0 = " 'reject h0' " if t_0eq > tcv_r_eq else " 'fail to reject h0' "   # 이건 되는 군. 
#     pval = 2* stats.t.sf( t_0eq, df_pool ) 

#     print(" 등분산 가정 t-검정  ") 
#     print(" t0, t_(a/2), 판단, pvalue : ", t_0eq , tcv_r_eq, yn_h0, pval) 

#     yn_h0 = " 'reject h0' " if t_0eq > tcv_r_un else " 'fail to reject h0' "   # 이건 되는 군. 
#     pval = 2* stats.t.sf( t_0un, df_hetr ) 

#     print(" 이분산 가정 t-검정  ") 
#     print(" t0, t_(a/2), 판단, pvalue : ", t_0un , tcv_r_un, yn_h0, pval) 

#     print(" 분산(등분산), 분산(이분산), 자유도(등), 자유도(이)")
#     print(var_eq, var_un, df_pool, df_hetr)

# #%%
# # 모듈 이용 scipy.stats.ttest_ind 
# # 두 그룹 평균 비교.
# # 이 모든 것을 간단 코드로 실현.
#     tstat, pvalu = ttest_ind(hgt_f, hgt_m, equal_var=True)  # 등분산 가정 평균 비교
#     print("equal variance, t0, pvalue, dof")
#     print(tstat, pvalu)

#     tstat, pvalu = ttest_ind(hgt_f, hgt_m, equal_var=False) # 이분산 가정 평균 비교 
#     print("not equal variance, t0, pvalue, dof")
#     print(tstat, pvalu)

#%%
# 4. 코드 체크
#  변수 특성, 배열 형식, 계산 완료 등.
#    print('Saved figure to', saved_path)
    print("Computing OK")

#%%
# 5. 코드 일괄 실행. __main__ is not main()  
# "이 파일이 직접 실행됐을 때만 이 코드를 돌려라, import돼서 불려올 땐 돌리지 마라"
# 이유 
# 파이썬은 어떤 파일이 실행될 때, 그 파일에 자동으로 __name__이라는 특수 변수를 생성.
# 이 변수 __name__ 에 들어가는 값:
# 그 파일을 직접 실행하면 → __name__에 "__main__"이라는 문자열 투입
# 그 파일을 다른 파일에서 import 하면 → __name__에 그 파일 이름 투입. thisisthefile.py라면 "thisisthefile"
if __name__ == "__main__":
    # "이 파일이 직접 실행됐을 때만 이 코드를 돌려라, import돼서 불려올 땐 돌리지 마라"
    # 아래 코드는 이 파일에서 직접 실행할 때 실행된다는 뜻.
    # 실행 전 설정
    input_dir = 'in_files' 
    output_dir = 'o_files'
    output_fig_dir = 'figures'

    input_file = 'cs_nns_gndr_hgt.xlsx' # 변수명: 체크 
    input_sheet = 'data'                     # Individual, Salary, Exper, Gender 
#    input_file = 'cs_prof_sal_exp_gndr.xlsx' # 변수명: 체크 
#    input_sheet = 'ISEG'                     # Individual, Salary, Exper, Gender 
    output_file = 'test'    # yet to be assigned 
    output_fig_file = '0test.png'

    seed = 12348215 
    alpha = 0.05                          # significance level, two side.

    main()

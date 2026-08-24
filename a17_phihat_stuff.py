#%%
# _rd_00.py 
# 분산 추론. 해보자.  
# 분산비 추론. 
# F비 
# raw data - 연봉, 경력 기간, 성별.
# _rd_00.py ==> 이건 벤치마크용.


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
from scipy.stats import ttest_1samp, f

#%%
# 분포 호출. 아래와 같은 방식. 
# norm.pdf, norm.cdf, norm.ppf, norm.sf, norm.isf

# def save_chart(fname, outdir, fig=None, dpi=300, bbox_inches='tight', transparent=False):
#     if fig is None:
#         fig = plt.gcf()
#     os.makedirs(outdir, exist_ok=True)
#     name, ext = os.path.splitext(fname)  # fname(파일)을 이름과 익스텐션으로 분리.
#     ext = ext or '.png'
#     path = os.path.join(outdir, name + ext)  # outdir에 fname을 짤라서 그림 이름으로 지정하네? 
#     fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
#     return path

#%%
def variance_ratio_test(x, y, alpha) : # alpha=0.05):
    x = np.asarray(x)  # 이게 버릇이군. 이하 ndarray로 변환/유지한다는 뜻. 일관성 유지.
    y = np.asarray(y)

    n1 = len(x)
    n2 = len(y)

    s1 = np.var(x, ddof=1)   # 자유도를 일상적으로 지정하자. 디폴트로 넘어가다가 미세 차이 발생 가능.
    s2 = np.var(y, ddof=1)

    F = s1 / s2
    df1 = n1 - 1
    df2 = n2 - 1
                             # 분포 호출 시, cdf, sf, ppf, isf 참조. 1-cdf보다는 sf. 
    p = 2 * min( f.cdf(F, df1, df2),
                 f.sf(F, df1, df2) ) 
              #  1 - f.cdf(F, df1, df2))  # 이것보다 f.sf를 쓰라는, 그게 안정적이라는.

    ci = (
        F / f.ppf(1 - alpha/2, df1, df2),
        F / f.ppf(alpha/2, df1, df2)
        )

    return F, p, ci
#%%
def main():
# 1. 데이터 준비. 읽기. 생성. 
#   데이터 읽고 처리 단계.
#   dat을 한 번에 읽었다, 3개 변수로 지정. dat은 data frame 형식.
    dat = pd.read_excel(os.path.join(input_dir, input_file), sheet_name=input_sheet) 
#   n_obs = len(dat) 
#   id = dat['i'].to_numpy()        # 그냥 번호. 별도 불요.
    gn = dat['gender'].to_numpy()   # 이렇게 하면 배열로 전환. dataframe -> NumPy 배열
    hgt = dat['ht'].to_numpy()
    # 문자인 경우 처리 방법 아이디어.
    # gndrlit = dat['Gender']                  # 문자 Male, Female. Pandas series 
    # gndr = (gndrlit == 'Male').astype(int)   # astype(int)으로 NumPy 배열 전환, 1/0  


# 2. groups: by gndr (gender, 1 - 0), 
    gndr = (gn == 1).astype(int)    # gn = 1, 2; gndr = 1, 0. 1/0으로 전환. 
    #hgt_ref = np.mean(hgt)          # 그룹 구분 참고값. 평균 이상, 이하.
    #hgt_grp = np.where(hgt > hgt_ref, "Tall", "Short")   # 기준 키 대비 Tall, Short.
    #dat['ht_g'] = hgt_grp            # data frame에 변수 추가된 것임. 
    dat['gndr'] = gndr               #   이것은 위에 1/0 자료를 추가. 성별임.

    hgt_m = hgt[gndr == 1]  # 성별 1, m 그룹 키. 그룹별 자료 분리, 배열은 hgt 하나임.
    hgt_f = hgt[gndr == 0]  #     0, f. 키 그룹별 자료 분리
    #mf_tall = dat[dat['ht_g'] == 'Tall']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  
    #mf_shrt = dat[dat['ht_g'] == 'Short']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  
    n_all = len(dat) 
    n_m = len(hgt_m)
    n_f = len(hgt_f)

# 분포 임계치, 양방향, 우측값. 적당한 위치 모색.
# 분산은 카이제곱 분포. 
# 임계치, 전체 
#    alpha = 0.05                          # significance level, two side.
#    kcv_r = stats.chi2.ppf(1-alpha/2, df_all ) # critical value on the right, two-side 
#    kcv_l = stats.chi2.ppf(alpha/2, df_all ) # critical value on the left,
#    kcv_r1 = stats.chi2.ppf(1-alpha, df_all ) # critical value on the right, one-side  
#    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
#    tcv_r = stats.t.ppf(1 - alpha / 2, df= n_tall)

# 3. 분산비: 성별 대비. male/female
# 추정치: 분산, overall, 불요
# overall 
    # sig2hat = hgt.var() 
    # df_all = n_all -1 
    # kcv_r = stats.chi2.ppf(1-alpha/2, df_all ) # critical value on the right, two-side 
    # kcv_l = stats.chi2.ppf(alpha/2, df_all ) # critical value on the left,

# 그룹별( m - f)
    sig2hat_m = hgt_m.var(ddof=1)  # ddof=1 <-- 표본분산; ddof=0 <-- 모분산, mle.
#    sig2hat_m = hgt_m.var()  # 이게 ndarray네. 디폴트 자유도 조정 0. 
                              # 판다스 시리즈아렴, 디폴트는 1.
                              # 자유도 때문에 미세한 차이 발생. 
    df_m = n_m -1 
    # kcv_rm = stats.chi2.ppf(1-alpha/2, df_m ) # critical value on the right, two-side 
    # kcv_lm = stats.chi2.ppf(alpha/2, df_m ) # critical value on the left,

    sig2hat_f = hgt_f.var(ddof=1) 
    df_f = n_f -1 
    # kcv_rf = stats.chi2.ppf(1-alpha/2, df_f ) # critical value on the right, two-side 
    # kcv_lf = stats.chi2.ppf(alpha/2, df_f ) # critical value on the left,

# 분산비 male/female 
    phi = sig2hat_m / sig2hat_f 
    fcv_r = stats.f.ppf( 1 - alpha/2, df_m, df_f) 
    fcv_l = stats.f.ppf( alpha/2, df_m, df_f) 


# 신뢰구간, right, left, 
# 그룹별 ( m - f )
    ci_r = phi / fcv_l  
    ci_l = phi / fcv_r
    print( "    phi,     confidence interval ")
    print( " F ratio:", phi,  ci_l, ci_r ) 


# 가설검정. 전체, 톨 
# 귀무가설 H0: mu = mu0
# 검정통계치, pvalue 

#    phi_zero = 1   # 등분산.

    f_0 =  phi / phi_zero   # F 검정통계량
    
    yn_h0 = " 'reject h0' " if (f_0 > fcv_r or f_0 < fcv_l)  else " 'fail to reject h0' "   # 이건 되는 군. 
    res = int( f_0 > fcv_r or f_0 < fcv_l)  # 이것은 기각 성공 1, 기각 실패 0
#    pval = 2*( 1 - stats.chi2.cdf( k_0, df_all ) ) 
    a = stats.f.cdf( f_0, df_m, df_f ) 
    b = stats.f.sf( f_0, df_m, df_f )    # 1- cdf , f.sf가 더 안정적이라는 거라고...
    pval = 2 * min(a, b)
    print(" 전체   : ", f_0 , fcv_l, fcv_r, yn_h0, pval) 

 

#   ==== 여기는 작업 중, 졸려서 스톱... 7.25.  진행 중.
# 모듈 이용 , 전용 모듈은 없다고 하네...
# F-검정 기반 분산 비율 신뢰구간 계산 (Homoscedasticity test)
    import pingouin as pg
#    res = pg.homoscedasticity(data=[hgt_m, hgt_f], method='levene')
    res = pg.homoscedasticity(data=[hgt_m, hgt_f], method='bartlett')
    print(res)
# chatgpt  # 이것은 직접 계산한 것과 같음. 모듈 아님. 분포를 불러오는 함수가 다양하군.
    lower = phi / f.ppf(1 - alpha/2, df_m, df_f )
    upper = phi / f.ppf(alpha/2, df_m, df_f )

    pvals = 2 * min(f.cdf( phi , df_m, df_f),
                   1 - f.cdf(phi , df_m, df_f)) 
    print("95% CI =", (lower, upper), pvals )
# chatgpt 제안 함수 설정. 
# 분산비 검정 함수 지정, 메인() 밖 별도.
    phit, pvalt, cit = variance_ratio_test( hgt_m, hgt_f, alpha )
    print(" phi, pval, ci ", phit, pvalt , cit )
    print(" hgt 판다 넘비", type(hgt_m) )
# 그런데, ci 계산 값에서 약간의 차이 발생. 소숫점 4자리 이하. 
# 왜? 같은 인공지능에, 같은 방식인데, 함수가 같거든. cdf 보른 거랑 ppf 부른 게 차이인가?
#  

# 2. 자료 검토, 요약, 그래프 생성.
# 2a. 자료 검토, 기초 통계량, 그룹별.
#   some = sal.describe()    # built-in pandas 
#    some = dat[["Salary", "Exper"]].describe()   # dat에서 선택된 변수의 기술통계 생성.
#    some = dat.describe()                         # dat에서 모든 양적 변수의 기술통계 생성.
#   dat.groupby("Gender")[["Salary", "Exper"]].describe()  # 이건 dataformat의 일부 변수만
#   dat.groupby("Gender").describe()                       # 이건 dataformat의 전체 변수 
#   dat.groupby("Gender").agg(["count", "mean", "min", "median", "max"]) # 전체 변수, 일부 통계량 
    # some = dat.groupby("Gender")[["Salary", "Exper"]].agg(
    #         ["count", "mean", "std", "min", "median", "max"]
    #         ) # 일부 변수, 일부 통계량 
    # print(some)

#    print( "covmat\n", covmat, "\nrhomat\n", rhomat)
#    print(f"covmat\n{covmat}\n\nrhomat\n{rhomat}") 

# 2d. 자료 검토, 시각화 확인. 
# 그룹별 상관계수, 공분산, 산포도, 작업... 
#     sal_g = {}              # 이게 딕셔너리, for에서 순서대로 받을 것을 미리 설정  
#     xpr_g = {}              # 뒤에 차례대로 쌓음을 표시 [g]
#     cov_g = {}
#     rho_g = {}
#     for g in gndr.unique():                  # for 문장이라...  for 아래는 반복되는 것...
#                                              # g는 gndr 값, 0, 1. 문자도 가능하다네.
#         sal_g[g] = sal[gndr==g]                 # nunpy 배열이니...
#         xpr_g[g] = xpr[gndr==g]
#  #       sal_g, xpr_g = sal[gndr==g], xpr[gndr==g]  # 위 두 줄을 한 줄로 나타낼 수 있군. 
#         covmat = np.cov(sal_g[g], xpr_g[g])
#         cov_g[g] = covmat[0, 1]
#         rhomat = np.corrcoef(sal_g[g], xpr_g[g])
#         rho_g[g] = rhomat[0, 1]
# #        print(g, covmat, rhomat)
#     # for g in gndr.unique():                 # for 문장, series 일 때... 표시 방식 차이...
#     #     dat_i = dat.loc[gndr == g, ['Salary', 'Exper']]
#     #    print(g, "\n", sal_g, "\n", xpr_g, "\n")
#         print(f"covmat\n{covmat}\n\nrhomat\n{rhomat}") 
#         print(g, cov_g[g], rho_g[g])   # for 문장이라...  for 아래는 반복되는 것... 들여 씀...
# #    print(sal_g. xpr_g)

# 그룹별 산도포 subplot 구성 
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # for i, g in enumerate(np.unique(gndr)):
    #     ax[i].scatter(sal_g[g], xpr_g[g])
    #     ax[i].set_title(f"Gender {g}")
    # plt.tight_layout()
    # plt.show()   # 화면에 보여줌, 저장은 아래 참조. 

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


#     # 엑셀로 출력. pandas DataFrame으로 변환 후 to_excel(), to_csv() 메소드 사용.
#     dfrm1 = pd.DataFrame(frequency_table1)
#     output_dir = 'o_files'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# #   dfrm.to_excel(os.path.join(output_dir, 'frequency_table.xlsx'), index=False)
#     dfrm1.to_csv(os.path.join(output_dir, 'frq_tbl1.csv'), index=False)
#%%
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
    output_file = 'test'    # yet to be assigned 
    input_dir = 'in_files' 
    input_file = 'cs_nns_gndr_hgt.xlsx' # 변수명: 체크 
    input_sheet = 'data'                     # Individual, Salary, Exper, Gender 
    seed = 12348215 
#    input_file = 'cs_prof_sal_exp_gndr.xlsx' # 변수명: 체크 
#    input_sheet = 'ISEG'                     # Individual, Salary, Exper, Gender 
    alpha = 0.05                          # significance level, two side.
    phi_zero = 1   # 등분산.


    main()

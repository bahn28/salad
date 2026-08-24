#%%
# _rd_00.py 
# 분산 추론. 해보자. 진행 중. 
# 분산의 분산, 카이제곱 이런 것들. K통계량. 
# var() -> ndarray 디폴트 0, dataframe 디폴트 1 <<== 미세한 차이 발생 이유. 

# 분산비는 별도.
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
from scipy.stats import ttest_1samp

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
    alpha = 0.05                          # significance level, two side.
#    kcv_r = stats.chi2.ppf(1-alpha/2, df_all ) # critical value on the right, two-side 
#    kcv_l = stats.chi2.ppf(alpha/2, df_all ) # critical value on the left,
#    kcv_r1 = stats.chi2.ppf(1-alpha, df_all ) # critical value on the right, one-side  
#    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
#    tcv_r = stats.t.ppf(1 - alpha / 2, df= n_tall)

# 3. 분산 추정치: 전체, 성별, 표준오차
# 추정치: 분산, overall
# overall 
    sig2hat = hgt.var(ddof=1) 
    df_all = n_all -1 
    kcv_r = stats.chi2.ppf(1-alpha/2, df_all ) # critical value on the right, two-side 
    kcv_l = stats.chi2.ppf(alpha/2, df_all ) # critical value on the left,


#    se_muhat = hgt.std() / np.sqrt( n_all )

# 그룹별( m - f)
    sig2hat_m = hgt_m.var(ddof=1) 
    df_m = n_m -1 
    kcv_rm = stats.chi2.ppf(1-alpha/2, df_m ) # critical value on the right, two-side 
    kcv_lm = stats.chi2.ppf(alpha/2, df_m ) # critical value on the left,

    sig2hat_f = hgt_f.var(ddof=1) 
    df_f = n_f -1 
    kcv_rf = stats.chi2.ppf(1-alpha/2, df_f ) # critical value on the right, two-side 
    kcv_lf = stats.chi2.ppf(alpha/2, df_f ) # critical value on the left,

# 신뢰구간, right, left, 
# overall 
    ci_r = df_all * sig2hat / kcv_l 
    ci_l = df_all * sig2hat / kcv_r 
    print( "    sig2hat,     confidence interval ")
    print( sig2hat,  ci_l, ci_r ) 

# 그룹별 ( m - f )
    ci_rm = df_m * sig2hat_m / kcv_lm 
    ci_lm = df_m * sig2hat_m / kcv_rm 
    print( "    sig2hat,     confidence interval ")
    print( sig2hat_m,  ci_lm, ci_rm ) 

    ci_rf = df_f * sig2hat_f / kcv_lf 
    ci_lf = df_f * sig2hat_f / kcv_rf 
    print( "    sig2hat,     confidence interval ")
    print( sig2hat_f,  ci_lf, ci_rf ) 


# 가설검정. 전체, 톨 
# 귀무가설 H0: mu = mu0
# 검정통계치, pvalue 

    sig2_zero = 40

    k_0 =  df_all * sig2hat / sig2_zero             # overall 
    k_0m = df_m * sig2hat_m / sig2_zero             # male
    k_0f = df_f * sig2hat_f / sig2_zero             # female

    yn_h0 = " 'reject h0' " if (k_0 > kcv_r or k_0< kcv_l)  else " 'fail to reject h0' "   # 이건 되는 군. 
    res = int(k_0 > kcv_r or k_0 < kcv_l)  # 이것은 기각 성공 1, 기각 실패 0
#    pval = 2*( 1 - stats.chi2.cdf( k_0, df_all ) ) 
    a = stats.chi2.cdf( k_0, df_all ) 
    b = stats.chi2.sf( k_0, df_all )    # 1- cdf 
    pval = 2 * min(a, b)
    print(" 전체   : ", k_0 , kcv_l, kcv_r, yn_h0, pval) 

# ***** 여기는 gender = 1, male 
    yn_h0m = " 'reject h0' " if (k_0m > kcv_rm or k_0m< kcv_lm)  else " 'fail to reject h0' "   # 이건 되는 군. 
    resm = int(k_0m > kcv_rm or k_0m < kcv_lm)  # 이것은 기각 성공 1, 기각 실패 0
    a = stats.chi2.cdf( k_0m, df_m ) 
    b = stats.chi2.sf( k_0m, df_m )    # 1- cdf 
    pvalm = 2 * min(a, b)
   #pvalm = 2*( 1 - stats.chi2.cdf( k_0m, df_m ) )
    print(" 전체m  : ", k_0m , kcv_lm, kcv_rm, yn_h0m, pvalm) 

 
 # ***** 여기는 gender = 0, female 
    yn_h0f = " 'reject h0' " if (k_0f > kcv_rf or k_0f< kcv_lf)  else " 'fail to reject h0' "   # 이건 되는 군. 
    resf = int(k_0f > kcv_rf or k_0f < kcv_lf)  # 이것은 기각 성공 1, 기각 실패 0
    a = stats.chi2.cdf( k_0f, df_f ) 
    b = stats.chi2.sf( k_0f, df_f )    # 1- cdf 
    pvalf = 2 * min(a, b)
    #pvalf = 2*( 1 - stats.chi2.cdf( k_0f, df_f ) )
    print(" 전체f  : ", k_0f , kcv_lf, kcv_rf, yn_h0f, pvalf) 

 

#   ==== 여기는 작업 중, 졸려서 스톱... 7.25.  진행 중.
# # 모듈 이용 statsmodels.stats.weightstats
# 분산에 대한 추론.
# 신뢰구간 
#  전체
#    from statsmodels.stats.weightstats import DescrStatsW
#    ds_hgt = DescrStatsW( hgt )
#    ci = ds_hgt.tconfint_mean(alpha=0.05)

# 평균 t 검정. 모듈이 여럿임.
#  모듈1, descrstatsw 모듈이네, 위에 있는 것, ci 
#    t0_a, pval_a, df_a = ds_hgt.ttest_mean(mu_zero) # mu_zero겠지? 
#    t, p, df = ds_hgt.ttest_mean(15) # mu_zero겠지? 

#  모듈2
#    from statsmodels.stats.weightstats import ztest
#    z, p = ztest( hgt, value= mu_zero )   # 요건 가설 평균 같은데.
#    z, p = ztest( hgt, value=15)   # 요건 가설 평균 같은데.

#  모듈3
#    from scipy.stats import ttest_1samp
#    t, p = ttest_1samp( hgt, popmean= mu_zero ) 
#    t, p = ttest_1samp( hgt, popmean=15)  # 분산 unknown 

#    print(t0_a , pval_a , df_a)


# 두 그룹 평균 비교는 다음에... 별도 코드. 파일.




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


    main()

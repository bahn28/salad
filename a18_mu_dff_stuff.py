# _rd_00.py 
# 평균차 추론. 평균차 분산, 표준오차, 신뢰구간, 가설검정  
#   등분산 전제, 이분산 전제.

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
from scipy.stats import ttest_1samp, ttest_ind, f


# def save_chart(fname, outdir, fig=None, dpi=300, bbox_inches='tight', transparent=False):
#     if fig is None:
#         fig = plt.gcf()
#     os.makedirs(outdir, exist_ok=True)
#     name, ext = os.path.splitext(fname)  # fname(파일)을 이름과 익스텐션으로 분리.
#     ext = ext or '.png'
#     path = os.path.join(outdir, name + ext)  # outdir에 fname을 짤라서 그림 이름으로 지정하네? 
#     fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
#     return path


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


def main():
# 1. 데이터 준비. 읽기. 생성. 
#   dat을 한 번에 읽었다, 3개 변수로 지정. dat은 data frame 형식.
    n_frac = 0.02   # 2%
    dat_all = pd.read_excel(os.path.join(input_dir, input_file), sheet_name=input_sheet) 
    n_obs = len(dat_all)
    n_sel = int(n_obs*n_frac)
#    dat = dat_all.sample(n_sel, replace=False, random_state=42)
    dat = ( dat_all
           .groupby('gender', group_keys=False)
           .apply(lambda x: x.sample(frac=n_frac, random_state=42))
    ) 

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

# 등분산 검정, 정의된 함수 이용.
    phi_hat, pval, ci = variance_ratio_test( hgt_m, hgt_f, alpha) 
    print(" 등분산 검정 :", phi_hat, pval)
    print(" 등분산 " if pval > alpha else " 이분산 ")

# 분포 임계치, 양방향, 우측값. 적당한 위치 모색.
#    alpha = 0.05                          # significance level, two side.
#    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
#    tcv_r = stats.t.ppf(1 - alpha / 2, df= n_tall)

# 3. 평균 차이 추정치 ( m - f ), 표준오차. 
# 추정치: 평균, 분산, 표준오차, overall
# overall 
    muhat = hgt.mean()
    var_hgt = hgt.var(ddof=1)  #/  n_all # 이게 통합 분산은 아니지. 

# 그룹별( m - f)
    muhat_m = hgt_m.mean()
    muhat_f = hgt_f.mean()
    var_hgt_m = hgt_m.var(ddof=1) # /  n_m 
    var_hgt_f = hgt_f.var(ddof=1) #/  n_f 

## 평균차의 분산, 표준오차 (등분산 vs. 이분산)
#   등분산 
    df_pool = n_m + n_f - 2                        # 등분산 자유도 
    sse_pool = (n_m-1) * var_hgt_m + (n_f-1) * var_hgt_f   # 등분산 가정 시 적용 가능 
    var_pool = sse_pool / df_pool                  # 공통분산 

    var_eq =  var_pool * ( 1 / n_m + 1 / n_f )     # 평균차 분산, 등분산인 경우. 
    se_eq = np.sqrt( var_eq )

#   이분산 
    var_un =  var_hgt_m / n_m + var_hgt_f / n_f   # 평균차이 분산, 이분산(일반) 
    se_un = np.sqrt( var_un )
    df_hetr =  var_un**2 / ( 
        (var_hgt_m/n_m)**2/(n_m-1 )
        + (var_hgt_f/n_f)**2/(n_f-1) ) # Welch–Satter 방법 
                                                   #   이분산 자유도 

# 평균차이
    mu_diff = muhat_m - muhat_f 
    print("평균차이:", mu_diff) 

# 신뢰구간, right, left, 등분산 
    tcv_r_eq = stats.t.ppf( 1- alpha/2, df_pool )  # critical value on the right, two side
    tcv_r_un = stats.t.ppf( 1- alpha/2, df_hetr )  # critical value on the right, two side

#    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side

    ci_r = mu_diff + tcv_r_eq * se_eq 
    ci_l = mu_diff - tcv_r_eq * se_eq   

    print( "등분산 가정:")
    print( "   mean diff,    se of mu_diff,     confidence interval ")
    print( mu_diff, se_eq, ci_l, ci_r ) 

# 신뢰구간, right, left, 이분산 --- 임계치와 표준오차 조정/변경/선택, 자유도...  
    ci_r = mu_diff + tcv_r_un * se_un 
    ci_l = mu_diff - tcv_r_un * se_un   

    print( "이분산 가정:")
    print( "HETR  mean diff,    se of mu_diff,     confidence interval ")
    print( mu_diff, se_un, ci_l, ci_r ) 


 
# 가설검정. 전체, 톨 
# 귀무가설 H0: mu = mu0
# 검정통계치, pvalue 

    print("H0: 평균이 동일하다")
    mu_diff_zero = 0

    t_0eq = np.abs( ( mu_diff - mu_diff_zero ) / se_eq  )       # 등분산 
    t_0un = np.abs( ( mu_diff - mu_diff_zero ) / se_un  )       # 이분산

    yn_h0 = " 'reject h0' " if t_0eq > tcv_r_eq else " 'fail to reject h0' "   # 이건 되는 군. 
    pval = 2* stats.t.sf( t_0eq, df_pool ) 

    print(" 등분산 가정 t-검정  ") 
    print(" t0, t_(a/2), 판단, pvalue : ", t_0eq , tcv_r_eq, yn_h0, pval) 

    yn_h0 = " 'reject h0' " if t_0eq > tcv_r_un else " 'fail to reject h0' "   # 이건 되는 군. 
    pval = 2* stats.t.sf( t_0un, df_hetr ) 

    print(" 이분산 가정 t-검정  ") 
    print(" t0, t_(a/2), 판단, pvalue : ", t_0un , tcv_r_un, yn_h0, pval) 

    print(" 분산(등분산), 분산(이분산), 자유도(등), 자유도(이)")
    print(var_eq, var_un, df_pool, df_hetr)

#   ==== 여기는 작업 중, 졸려서 스톱... 7.25.  진행 중.
# # 모듈 이용 statsmodels.stats.weightstats
# 평균에 대한 추론.
# 신뢰구간 
#  전체
#    from statsmodels.stats.weightstats import DescrStatsW
    # ds_hgt = DescrStatsW( hgt )
    # ci = ds_hgt.tconfint_mean(alpha=0.05)

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
    tstat, pvalu = ttest_ind(hgt_f, hgt_m, equal_var=True)  # 등분산 가정 평균 비교
    print("equal variance, t0, pvalue, dof")
    print(tstat, pvalu)

    tstat, pvalu = ttest_ind(hgt_f, hgt_m, equal_var=False) # 이분산 가정 평균 비교 
    print("not equal variance, t0, pvalue, dof")
    print(tstat, pvalu)




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

    main()

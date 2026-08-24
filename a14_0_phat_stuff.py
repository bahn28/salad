#%%
# _rd_00.py 
# 비율 추론. 해보자. phat, se, 신뢰구간, 검정통계량, pvalue.  
# 두 비율의 비교, 진행 중.
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

#%%

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


# 2. groups: by gndr (gender, 1 - 0), by hgt_grp (height, Tall - Short) 
    gndr = (gn == 1).astype(int)    # gn = 1, 2; gndr = 1, 0. 1/0으로 전환. 
    hgt_ref = np.mean(hgt)          # 그룹 구분 참고값. 평균 이상, 이하.
    hgt_grp = np.where(hgt > hgt_ref, "Tall", "Short")   # 기준 키 대비 Tall, Short.
    dat['ht_g'] = hgt_grp            # data frame에 변수 추가된 것임. 
    dat['gndr'] = gndr               #   이것은 위에 1/0 자료를 추가. 성별임.

    hgt_tall = hgt[hgt_grp=='Tall']  # 키 그룹별 자료 분리, 배열은 hgt 하나임.
    hgt_shrt = hgt[hgt_grp=='Short'] # 키 그룹별 자료 분리
    mf_tall = dat[dat['ht_g'] == 'Tall']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  
    mf_shrt = dat[dat['ht_g'] == 'Short']['gndr']  # Tall 그룹 성별을 1/0으로 받음.  
    n_all = len(dat) 
    n_tall = len(hgt_tall)
    n_shrt = len(hgt_shrt)
#   mf_tall = hgt[hgt_grp=='Tall']['gender']  # 키 그룹별 자료 분리, 오류. 변수 생성 순서 문제? 
#   mf_tall = dat[dat['ht'] > hgt_ref ]['gender']  # 그렇구만. 딩동
#   mf_tall = dat[dat['ht_g'] == 'Tall']['gender']  # 그렇구만. 딩동, 큰키 그룹의 성별, 1/2, 이건. 원래.

# 분포 임계치, 양방향, 우측값. 적당한 위치 모색.
    alpha = 0.05                          # significance level, two side.
    zcv_r = stats.norm.ppf( 1- alpha/2 )  # critical value on the right, two side
#    tcv_r = stats.t.ppf(1 - alpha / 2, df= n_tall)

# 3. 성 비율 추정치, 표준오차 (전체)
# 추정치: 비율, 비율의 표준오차, overall
    phat_m = (gndr== 1).mean()                # 이렇게 해도 되네... gndr = 1인 그룹 비율.
#    phat_m = np.sum(gndr)/len(gndr)          # len(gndr)은 n_all과 같음. 
#    phat_f = 1 - phat_m
    se_phat_m = np.sqrt(phat_m * (1 - phat_m) / n_all )  # 옵션 1임.
    se_max = np.sqrt( 0.5 * (1 - 0.5 ) / n_all )         # 옵션 2임. 최대 표준오차.

# 신뢰구간, right, left, (옵션 1).
    ci_r = phat_m + zcv_r * se_phat_m 
    ci_l = phat_m - zcv_r * se_phat_m 

    print( "    phat,    se of phat,    se max,    confidence interval ")
    print( phat_m, se_phat_m, se_max,  ci_l, ci_r ) 

# 추정치(그룹별): 비율, 비율의 표준오차, Tall - Short group, by height 

    phat_mt = mf_tall.mean()   # 이것도 되네...  위, Tall group의 성별임. mf.  
#    phat_mt = (gndr== 1)[hgt_grp=='Tall'].mean()      # 이렇게 해도 되네...
#    phat_mt = np.sum(gndr[hgt_grp=='Tall'])/len(hgt_tall)
#    phat_ft = 1 - phat_mt  
    se_phat_mt = np.sqrt(phat_mt * (1 - phat_mt) / n_tall )  # 옵션 1임.
    ci_rt = phat_mt + zcv_r * se_phat_mt 
    ci_lt = phat_mt - zcv_r * se_phat_mt 

    print("Tall (옵션1)", phat_mt, se_phat_mt, ci_lt, ci_rt ) 

# 가설검정. 전체, 톨 
# 귀무가설 H0: p=p0
# 검정통계치, pvalue 

    p_zero = 0.45
#    se_zero = np.sqrt( p_zero * (1 - p_zero) / n_all)   # 옵션 3.
#    t_0 = np.abs( ( phat_m - p_zero ) / se_zero )       # 옵션 3.
    t_0m = np.abs( ( phat_m - p_zero ) / se_phat_m )       # 옵션 1.
    yn_h0m = " 'reject h0' " if t_0m > zcv_r else " 'fail to reject h0' "   # 이건 되는 군. 
#    yn_h0 = ["accept", "reject"][ t_0 > zc_up ]   # 거짓일 때, 참일 때, 안 되는 군. 
    pvalm = 2*( 1 - stats.norm.cdf( t_0m ) )

    print(" 전체 (옵션1)", t_0m , zcv_r, yn_h0m, pvalm) 

# ***** 여기는 Tall 
    t_0mt = np.abs( ( phat_mt - p_zero ) / se_phat_mt )       # 옵션 1.

    yn_h0 = " 'reject h0' " if t_0mt > zcv_r else " 'fail to reject h0' "   # 이건 되는 군. 
#    yn_h0 = ["accept", "reject"][ t_0 > zc_up ]   # 거짓일 때, 참일 때, 안 되는 군. 
    pvalmt = 2*( 1 - stats.norm.cdf( t_0mt ) )

    print(" Tall (옵션1)", t_0mt , zcv_r, yn_h0, pvalmt) 
   
# 모듈 이용 statsmodels.stats.proportion 
    count_mf = gndr.sum()          # number of male, male=1, female=0, overall, 
    count_mf_tall = mf_tall.sum()  # number of male among tall group, male=1, female=0
#    n_tall = len(mf_tall)    # 처음에 있음

    count_mf_shrt = mf_shrt.sum()  # number of male among shrt group, male=1, female=0
#    n_shrt = len(mf_shrt)
    ci_all = proportion_confint(count_mf, n_all, alpha=0.05, method="normal")
    ci_tall = proportion_confint(count_mf_tall, n_tall, alpha=0.05, method="normal")
    ci_shrt = proportion_confint(count_mf_shrt, n_shrt, alpha=0.05, method="normal")
    print(f"전체 그룹 1의 비율: {count_mf / n_all:.4f} (95% CI: {ci_all})")
    print(f"Tall 그룹 1의 비율: {count_mf_tall / n_tall:.4f} (95% CI: {ci_tall})")
    print(f"Shrt 그룹 1의 비율: {count_mf_shrt / n_shrt:.4f} (95% CI: {ci_shrt})")
#    print(hgt_ref, len(dat) , len(hgt_tall), len(hgt_shrt))
#    print(" overall \n ", phat_m, se_phat_m, 
#             " \n by group tall \n ", phat_mt, se_phat_mt 
#             ) 

#    z0, pv = proportions_ztest(count_mf_tall, n_tall, p_zero, 'two-sided', p_zero) 
    z0, pv = proportions_ztest(count_mf_tall, n_tall, p_zero, 'two-sided') 
            # 옵션1 
    print("\n모듈 결과: statsmodels.stats.proportion ")
    print(f"Tall, 그룹 1의 비율: {count_mf_tall / n_tall:.4f} (95% CI: {ci_tall})")
    print(" 검통 (옵1): ", z0, " p값 ", pv)




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

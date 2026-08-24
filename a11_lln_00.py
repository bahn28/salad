#%%
# _lln_00.py 
# law of large number, 오차의 크기
#   00은 베이스 코드
#   01 등은 정리, 업데이트 등.


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
# 1. 데이터 준비. 읽기. 생성. random number 
    rng = np.random.default_rng(seed)

    muhat_rslt = []   # 표본 크기에서 모의실험 결과 정리. 표본크기, 평균의 평균, 평균의 표준편차  
    for k in np.arange(size_beg, size_end, size_stp): # a에서 b까지 c씩 증가, 끝은 포함하지 않음. n_pop로 받자.
        n_pop=k 
        muhat = []   # 표본 크기 고정. 평균 관측치 묶음. 234회 => 평균치 234개, n_itr 개 
        for i in np.arange(0, n_itr, 1) :  # 0에서 itr까지 1씩 증가
        #for i in np.linspace(0, 1, n_itr) :
        #    x = rng.normal(loc=mu, scale=sig, size=n_pop)  # obs tbg 
            x = rng.uniform(x_beg, x_end, n_pop)
        #muhat = np.mean(x)
        #print(muhat)          # 매회 계산된 muhat 인쇄. 1개. n_pop 회 반복 
            muhat.append(np.mean(x)) # 매회 muhat 계산, 반복할 때 추가, append. 
                                     # 반복 끝나면 배열 n_itr*1인 셈. 
        #print("muhat \n", muhat)    # for 문 밖에서 한번에 인쇄 
                                 # append 없으면 for 반복에서 마지막 것만 기억. 이전 것은 덮어쓰기로 사라짐. 
        #    print("i", i) 
        #    히스토그램은 k가 주어진 상태에서 반복횟수만큼 계산된 평균치의 분포. 
        #    다음은 k가 달라질 때, 위 평균치의 평균이 보여주는 추세. 
        #plt.boxplot(muhat)  # n=200, n_itr 반복시 평균 분포, 상자그림, 히스토그램
        plt.hist(muhat)  # n=200, n_itr 반복시 평균 분포, 상자그림, 히스토그램 
                         # 이것을 n_pop 별로 묶어서 하나로 그리자. 4개 정도 어때. ?
        plt.show()
        muhat_mean = np.mean(muhat)     # 표본 크기 고정, n_itr 반복 평균치의 평균. for i 바깥임. 끝난 후. 
        muhat_std = np.std(muhat)
#        print(muhat_mean, muhat_std, len(muhat), n_itr, n_pop)   # for k 내부, for i 외부 
        
#        print("n pop ", k, n_pop)
        muhat_rslt.append( [n_pop, muhat_mean, muhat_std] )  # 표본 크기마다 계산된 통계치 쌓음. 
                                                             # n의 크기와 평균의 평균, 평균의 표준편차  
                                                             # 최종 산출물, 관심 대상.
#    print("muhat repeated over n\n", muhat_rslt, len(muhat_rslt) )
    muhat_rslt = np.array(muhat_rslt)   # numpy 배열로 전환. 
    print(muhat_rslt)                   # 포맷, 배열 형식 차이 확인... 

# 다블 반복. 아이디어.  
# 표본크기 100, 200, 400, 이런 식으로...  k, n_pop으로 받음
# 표본 크기 100일 때, 250회 실험. i, 1,2,... 이렇게 반복 인덱스임. 계산에 투입되는 것은 아님.  
#    평균(통계치) 계산 (매 반복).
#    반복 마다 평균(통계치) 달라짐, 변화, 즉 변수,
#    250회 반복 => 총 250개의 통계치 계산. ==> 평균은 분포를 보임.  
#    250회 반복 마지막, 250개 결과값 배열, 반복 루프 밖에 묶음 배열 반환.
#    루프 밖에서 250 결과값의 평균(평균의 평균), 표준편차(평균의 표준편차/표준오차) <== 관심 결과값.   
# 표본 크기 200 일 때, 위 실험 반복. 
#    위 관심 결과값 채집. 추가. append. 
# 표본 크기 변화해 가며 관심결과 채짐, 추가, append.
# 표본 크기 루프 마치면, 루푸 밖에 관심 결과값 배열 반환.
#    반환된 배열로 확인 시도. 
#    표본 크기와 (평균의 평균, 평균의 표준편차.표준오차)가 어떤 관계일까.
#      크면, 표준편차가 작아진다는 것을 보여라.
# 그래야 평균이 모평균에 가까워진다는 것을 보인다. 
# 그 때 히스토그램 변화 모습을 보여준다.  
    
    # plt.scatter(muhat_rslt.iloc[:,0], muhat_rslt.iloc[:,1])  # data frame에서 열 지정이네. 
    # plt.plot(muhat_rslt.iloc[:,0], muhat_rslt.iloc[:,1])
#    plt.scatter(muhat_rslt[:,0], muhat_rslt[:,1])  # numpy 배열에서 열 지정. 
#    plt.ticklabel_format(axis='y', style='plain', useOffset=False)
#    plt.show()

#    plt.plot(muhat_rslt[:,0], muhat_rslt[:,1])
#    plt.show()

# 5. 저장
#  저장 파일 이름 지정
    # saved_path = save_chart(
    #     fname = output_fig_file,  # 메인() 밖에서 지정 'sta_09001_corr_01.png'
    #     outdir = output_fig_dir, # 메인() 밖에서 지정 'figures', 
    #     fig=fig, 
    #     dpi=300)

# 6. 코드 체크
#  변수 특성, 배열 형식, 계산 완료 등.
    #print(saved_path) # for checking 
    #      figures\sta_09001_corr_01.png <-- 이게 saved_path 야. 
    #      output_fig_dir + output_fig_file 결합이군. 
#    print('Saved figure to', saved_path)
    print("Computing OK")
#    plt.show()


#%%
# 7. 메인() 일괄 실행. 
if __name__ == "__main__":
    # 실행 전 설정
    # 미리 준비하면서 정보를 파악해 두어야 함.
    # 아웃풋 폴더
    # 데이터 폴더, 데이터 파일 이름, 데이터 시트 이름,
    # 데이터의 그룹, 층, 범주 이름 등
    # 난수생성관련 조건 등 
    output_fig_dir = 'figures'
    output_fig_file = 'sta_11001_lln_01.png'
    output_dir = 'o_files'
    input_dir = 'in_files' 
    input_file = 'cs_nns_hgt_wgt.xlsx' # 변수명: x, y 
    input_sheet = 'hwght'
    strat = 'gender'  # stratum variable, check with data file, 2그룹인 경우.
    rrank = 'rnrk' # 이런 것은 미리 알고 있어야... 데이터 파악 단계에서. 
#    n_pop = 123    # 연습용 모집단 크기, 모의실험 대상.
    size_beg = 1000 # 표본 크기 지정, beg부터 end까지 stp 간격.
    size_end = 50000 
    size_stp = 10000 
    x_beg = 50       # 균등분포 시작, 끝 
    x_end = 150 
    n_itr = 356   # number of iterations, monte carlo simulation
    seed = 135678942 
    mu = 100 
    sig = 15 

    main()

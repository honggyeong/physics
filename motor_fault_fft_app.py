import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform

# 머신러닝 관련
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic)
if platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="VVVF 구동음 FFT 이상 감지 시뮬레이터", layout="wide")
st.title("VVVF 구동음 FFT 기반 전동기 이상 감지 시뮬레이션 (머신러닝 포함)")
st.markdown("""
- **1단계:** 정상/이상 신호의 파라미터를 조절하세요.
- **2단계:** 'FFT 분석 및 이상 감지' 버튼을 누르면 스펙트럼과 감지 결과가 표시됩니다.
- **3단계:** '머신러닝 분류' 버튼을 누르면 SVM 기반 자동 분류 결과가 표시됩니다.
""")

# 신호 파라미터 입력
col1, col2 = st.columns([1,1])
with col1:
    st.subheader('신호 설정')
    fs = st.number_input('샘플링 주파수 (Hz)', value=10000)
    duration = st.number_input('신호 길이 (초)', value=1.0)
    f1 = st.number_input('기본 주파수(Hz)', value=400)
    f2 = st.number_input('고조파(Hz)', value=800)
    f_fault = st.number_input('이상음(Hz)', value=1200)
    amp1 = st.slider('기본음 진폭', 0.0, 2.0, 1.0, 0.05)
    amp2 = st.slider('고조파 진폭', 0.0, 2.0, 0.5, 0.05)
    amp_fault = st.slider('이상음 진폭', 0.0, 2.0, 0.3, 0.05)
    noise = st.slider('노이즈(랜덤)', 0.0, 1.0, 0.05, 0.01)
    do_fault = st.checkbox('이상(결함) 신호 포함', value=True)
    analyze = st.button('FFT 분석 및 이상 감지')
    ml_run = st.button('머신러닝 분류')

# 신호 생성 함수
def make_signal(fs, duration, f1, f2, f_fault, amp1, amp2, amp_fault, noise, do_fault):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    normal = amp1*np.sin(2*np.pi*f1*t) + amp2*np.sin(2*np.pi*f2*t)
    if do_fault:
        signal = normal + amp_fault*np.sin(2*np.pi*f_fault*t)
    else:
        signal = normal
    signal += noise * np.random.randn(len(t))
    return t, signal

# FFT 분석 함수
def analyze_fft(signal, fs):
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq[:len(freq)//2], np.abs(fft)[:len(fft)//2]

with col2:
    if analyze:
        t, signal = make_signal(fs, duration, f1, f2, f_fault, amp1, amp2, amp_fault, noise, do_fault)
        freq, amp = analyze_fft(signal, fs)
        # 이상음 감지(임계값 기반)
        fault_idx = np.argmin(np.abs(freq - f_fault))
        fault_amp = amp[fault_idx]
        threshold = 0.15  # 임계값(조정 가능)
        is_fault = fault_amp > threshold
        # 그래프
        fig, ax = plt.subplots(2,1,figsize=(8,6))
        ax[0].plot(t, signal)
        ax[0].set_title('구동음 신호(시간 영역)')
        ax[0].set_xlabel('시간 (s)')
        ax[0].set_ylabel('진폭')
        ax[1].plot(freq, amp)
        ax[1].set_title('FFT 스펙트럼')
        ax[1].set_xlabel('주파수 (Hz)')
        ax[1].set_ylabel('진폭')
        ax[1].axvline(f_fault, color='r', linestyle='--', label='이상음 주파수')
        ax[1].legend()
        st.pyplot(fig)
        # 결과
        st.subheader('이상 감지 결과')
        if is_fault:
            st.error(f'이상음({f_fault} Hz) 성분이 감지되었습니다! (진폭: {fault_amp:.2f})')
        else:
            st.success(f'이상음({f_fault} Hz) 성분이 감지되지 않았습니다. (진폭: {fault_amp:.2f})')
    elif ml_run:
        st.subheader('머신러닝 기반 자동 분류')
        # 데이터셋 생성
        N = 100  # 샘플 수
        X = []
        y = []
        for _ in range(N//2):
            _, sig = make_signal(fs, duration, f1, f2, f_fault, amp1, amp2, 0.0, noise, False)
            _, amp_spec = analyze_fft(sig, fs)
            X.append(amp_spec[:200])  # 앞부분만 특징으로 사용
            y.append(0)  # 정상
        for _ in range(N//2):
            _, sig = make_signal(fs, duration, f1, f2, f_fault, amp1, amp2, amp_fault, noise, True)
            _, amp_spec = analyze_fft(sig, fs)
            X.append(amp_spec[:200])
            y.append(1)  # 이상
        X = np.array(X)
        y = np.array(y)
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # SVM 분류기
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write(f'정확도: {acc*100:.2f}%')
        st.write('혼동 행렬(Confusion Matrix):')
        st.write(cm)
        st.info('0=정상, 1=이상. FFT 스펙트럼만으로도 머신러닝이 정상/이상 신호를 높은 정확도로 분류할 수 있습니다.')
    else:
        st.write('신호 파라미터를 설정하고 버튼을 눌러 분석하세요.') 
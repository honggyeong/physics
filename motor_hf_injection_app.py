import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic)
import platform
if platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="센서리스 모터 초기 위치 추출 시뮬레이터", layout="wide")
st.title("고주파 주입을 통한 센서리스 모터 초기 위치 추출 시뮬레이션")
st.markdown("""
- **1단계:** 회전자 초기 위치(각도)를 설정하세요.
- **2단계:** '고주파 주입 및 위치 추출' 버튼을 누르면 고주파 주입 및 위치 추정이 시뮬레이션됩니다.
""")

# 파라미터
R = 1.0
L_d = 0.01
L_q = 0.015
V_hf = 10
f_hf = 2000
T = 0.01
fs = 100_000

# 사용자 입력: 회전자 위치
col1, col2 = st.columns([1,2])
with col1:
    theta_deg = st.slider("회전자 초기 위치 (도)", 0, 359, 45)
    detect = st.button("고주파 주입 및 위치 추출")

# 전동기 단면 시각화
def plot_motor_cross_section(theta_deg):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal')
    # 고정자
    stator = plt.Circle((0,0), 1, fill=False, lw=3, color='gray')
    ax.add_patch(stator)
    # d축 (0도)
    ax.arrow(0, 0, 0.8, 0, head_width=0.07, head_length=0.1, fc='b', ec='b', label='d축')
    ax.text(0.9, 0, 'd축', color='b', fontsize=14, ha='center', va='center')
    # q축 (90도)
    ax.arrow(0, 0, 0, 0.8, head_width=0.07, head_length=0.1, fc='g', ec='g', label='q축')
    ax.text(0, 0.9, 'q축', color='g', fontsize=14, ha='center', va='center')
    # 회전자 위치
    theta = np.deg2rad(theta_deg)
    ax.arrow(0, 0, 0.7*np.cos(theta), 0.7*np.sin(theta), head_width=0.09, head_length=0.12, fc='r', ec='r', label='회전자')
    ax.text(0.8*np.cos(theta), 0.8*np.sin(theta), f'{theta_deg}°', color='r', fontsize=14, ha='center', va='center')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    return fig

with col1:
    st.pyplot(plot_motor_cross_section(theta_deg))

# 고주파 주입 시뮬레이션 및 위치 추정
def simulate_hf_injection(theta_deg):
    t = np.linspace(0, T, int(fs*T))
    theta = np.deg2rad(theta_deg)
    # 살리엔시: L = Ld*cos^2 + Lq*sin^2
    L = L_d * np.cos(theta)**2 + L_q * np.sin(theta)**2
    # 전류 응답
    Z = np.sqrt(R**2 + (2*np.pi*f_hf*L)**2)
    i = V_hf / Z * np.sin(2*np.pi*f_hf*t)
    # 위치 추정: 가능한 각도별 진폭 비교
    test_angles = np.linspace(0, 2*np.pi, 360)
    amps = []
    for th in test_angles:
        L_test = L_d * np.cos(th)**2 + L_q * np.sin(th)**2
        Z_test = np.sqrt(R**2 + (2*np.pi*f_hf*L_test)**2)
        amps.append(V_hf / Z_test)
    est_idx = np.argmin(np.abs(np.max(i) - np.array(amps)))
    est_deg = int(np.rad2deg(test_angles[est_idx]))
    return t, i, est_deg

with col2:
    if detect:
        t, i, est_deg = simulate_hf_injection(theta_deg)
        st.subheader(f"추정된 회전자 위치: {est_deg}°")
        fig2, ax2 = plt.subplots(figsize=(7,3))
        ax2.plot(t*1000, i, label='전류 응답')
        ax2.set_xlabel('시간 (ms)')
        ax2.set_ylabel('전류 (A)')
        ax2.set_title('고주파 주입 전류 응답')
        ax2.legend()
        st.pyplot(fig2)
        st.info(f"실제 위치: {theta_deg}°, 추정: {est_deg}° (오차: {abs(theta_deg-est_deg)}°)")
    else:
        st.write(":arrow_left: 회전자 위치를 설정하고 버튼을 눌러 시뮬레이션하세요.") 
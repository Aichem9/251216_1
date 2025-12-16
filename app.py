import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 기본 레이아웃
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sea Ice AI Analyst",
    page_icon="🧊",
    layout="wide"
)

st.title("🧊 AI Sea Ice Analyst")
st.markdown("""
이 대시보드는 해빙(Sea Ice) 데이터를 분석하고, **OpenAI API**를 활용하여 
기후 변화 전문가의 관점에서 인사이트를 도출합니다.
""")

# -----------------------------------------------------------------------------
# 2. 사이드바 설정 (파일 업로드 & API 키)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정 (Settings)")

# API 키 입력 (보안을 위해 비밀번호 형태로 입력)
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (seaice.csv)", type=["csv"])

# -----------------------------------------------------------------------------
# 3. 데이터 로드 및 전처리 함수
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # 컬럼 공백 제거
    df.columns = df.columns.str.strip()
    # 날짜 컬럼 생성
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    return df

# -----------------------------------------------------------------------------
# 4. 메인 로직
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # --- 데이터 미리보기 ---
    with st.expander("📊 원본 데이터 미리보기"):
        st.dataframe(df.head())

    # --- 메인 대시보드 (탭 구성) ---
    tab1, tab2, tab3 = st.tabs(["📈 시계열 분석", "📅 연도별 추세", "🤖 AI 전문가 분석"])

    # [Tab 1] 전체 시계열 그래프
    with tab1:
        st.subheader("남반구 vs 북반구 해빙 면적 변화 (Daily)")
        fig_ts = px.line(
            df, x='Date', y='Extent', color='hemisphere',
            title='Daily Sea Ice Extent Over Time',
            labels={'Extent': 'Extent (10^6 sq km)'},
            template="plotly_white"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # [Tab 2] 연도별 평균 추세
    with tab2:
        st.subheader("연도별 평균 해빙 면적 추세")
        yearly_df = df.groupby(['Year', 'hemisphere'])['Extent'].mean().reset_index()
        
        fig_trend = px.scatter(
            yearly_df, x='Year', y='Extent', color='hemisphere',
            trendline="ols", # 추세선 추가
            title='Yearly Average Sea Ice Extent Trend',
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # [Tab 3] AI 분석 (OpenAI API 연동)
    with tab3:
        st.subheader("🤖 AI 환경 데이터 분석 리포트")
        
        if not api_key:
            st.warning("⚠️ 분석을 시작하려면 사이드바에 OpenAI API Key를 입력해주세요.")
        else:
            # AI에게 보낼 요약 통계 데이터 생성
            stats_north = df[df['hemisphere'] == 'north']['Extent'].describe().to_string()
            stats_south = df[df['hemisphere'] == 'south']['Extent'].describe().to_string()
            
            # 최근 5년 vs 초기 5년 비교 데이터 계산
            recent_years = df['Year'].max()
            start_years = df['Year'].min()
            
            recent_avg_n = df[(df['hemisphere']=='north') & (df['Year'] >= recent_years-5)]['Extent'].mean()
            past_avg_n = df[(df['hemisphere']=='north') & (df['Year'] <= start_years+5)]['Extent'].mean()
            
            # 프롬프트 구성
            system_prompt = "당신은 저명한 기후 과학자이자 데이터 분석가입니다. 주어진 데이터를 바탕으로 명확하고 통찰력 있는 분석 보고서를 한국어로 작성해야 합니다."
            
            user_prompt = f"""
            다음은 1978년부터 2019년까지의 해빙(Sea Ice) 면적 데이터 요약입니다.
            
            [데이터 요약]
            1. 북반구(North) 기초 통계:
            {stats_north}
            
            2. 남반구(South) 기초 통계:
            {stats_south}
            
            3. 북반구 변화 추이:
            - 초기 5년 평균: {past_avg_n:.2f}
            - 최근 5년 평균: {recent_avg_n:.2f}
            
            [요청 사항]
            이 데이터를 바탕으로 다음 내용을 포함한 분석 리포트를 작성해주세요:
            1. **전반적인 추세 요약**: 북반구와 남반구의 차이점
            2. **기후 변화의 영향**: 북반구 데이터 감소가 의미하는 바
            3. **데이터의 변동성**: 최대/최소 격차에 대한 해석
            4. **결론 및 제언**
            
            전문적인 용어를 사용하되, 비전문가도 이해하기 쉽게 설명해주세요. 마크다운 형식으로 출력하세요.
            """
            
            if st.button("🚀 AI 분석 실행하기"):
                try:
                    client = OpenAI(api_key=api_key)
                    
                    with st.spinner("AI가 데이터를 분석하고 있습니다... (약 10~20초 소요)"):
                        response = client.chat.completions.create(
                            model="gpt-4o",  # 또는 gpt-3.5-turbo
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.7
                        )
                        
                    analysis_text = response.choices[0].message.content
                    st.markdown(analysis_text)
                    
                except Exception as e:
                    st.error(f"에러가 발생했습니다: {e}")

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드해주세요.")

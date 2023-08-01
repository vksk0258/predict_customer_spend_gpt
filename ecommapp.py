# %%
import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import *
import json
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib

def my_function(df, input_value):
    llm = OpenAI(api_token=st.secrets["openai_key"])
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=input_value)
    return result


# %%
# Create a session to Snowflake with credentials
with open("connection copy.json") as f:
    connection_parameters = json.load(f)
session = Session.builder.configs(connection_parameters).create()

st.set_page_config(layout="wide")


st.markdown(
    """
<style>
    [data-testid="stMetricValue"] {
        font-size: 37px;
        color: #000080;
        font-weight : bold;
        text-align: center;
    }
""",
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
     <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
    }
    </style>
    """, unsafe_allow_html=True)



# %%
# Header
empty1, head1, empty2= st.columns([3, 9.7, 5])

with head1:
    st.write("<h5 style='font-size: 32px; font-weight : bold; text-align: center;'>쇼핑몰 고객 연간 소비액 예측 모델</h5>", unsafe_allow_html=True)

# %%
# Customer Spend Slider Column
empty1, col1, empty4, col2,empty3, col3, empty2 = st.columns([3, 2.7, 0.2, 2.7, 0.5, 5.6, 3])

customer_df = session.table('PREDICTED_CUSTOMER_SPEND')

# Read Data
minasl, maxasl, mintoa, maxtoa, mintow, maxtow, minlom, maxlom = customer_df.select(
    floor(min(col("SESSION_LENGTH"))),
    ceil(max(col("SESSION_LENGTH"))),
    floor(min(col("TIME_ON_APP"))),
    ceil(max(col("TIME_ON_APP"))),
    floor(min(col("TIME_ON_WEBSITE"))),
    ceil(max(col("TIME_ON_WEBSITE"))),
    floor(min(col("LENGTH_OF_MEMBERSHIP"))),
    ceil(max(col("LENGTH_OF_MEMBERSHIP")))
).toPandas().iloc[0, ]

minasl = int(minasl)
maxasl = int(maxasl)
mintoa = int(mintoa)
maxtoa = int(maxtoa)
mintow = int(mintow)
maxtow = int(maxtow)
minlom = int(minlom)
maxlom = int(maxlom)


# Column 1
with col1:
    st.write("<h2>Search</h2>", unsafe_allow_html=True)
    st.write("<h6>Web 평균 이용 시간 (분)</h6>", unsafe_allow_html=True)
    tow = st.slider("Time on Website", mintow, maxtow, (mintow, mintow+5), 1,label_visibility="collapsed")
    st.write("<h6>App 평균 이용 시간 (분)</h6>", unsafe_allow_html=True)
    toa = st.slider("Time on App", mintoa, maxtoa, (mintoa, mintoa+5), 1,label_visibility="collapsed")

    
with col2:
    st.markdown(" ")
    st.markdown("######")
    st.markdown("######")
    st.markdown("######")
    st.markdown("######")

    st.write("<h6>매장 평균 이용 시간 (분)</h6>", unsafe_allow_html=True)
    asl = st.slider("Session Length", minasl, maxasl, (minasl, minasl+5), 1,label_visibility="collapsed")
    st.write("<h6>맴버쉽 가입 년 수</h6>", unsafe_allow_html=True)
    lom = st.slider("Length of Membership", minlom,
                    maxlom, (minlom, minlom+4), 1,label_visibility="collapsed")
    
# Column 2 (3)
with col3:
    st.markdown("## Predict")

    try:
        minspend, maxspend = customer_df.filter(
            (col("SESSION_LENGTH") <= asl[1]) & (
                col("SESSION_LENGTH") > asl[0])
            & (col("TIME_ON_APP") <= toa[1]) & (col("TIME_ON_APP") > toa[0])
            & (col("TIME_ON_WEBSITE") <= tow[1]) & (col("TIME_ON_WEBSITE") > tow[0])
            & (col("LENGTH_OF_MEMBERSHIP") <= lom[1]) & (col("LENGTH_OF_MEMBERSHIP") > lom[0])
        ).select(trunc(min(col('PREDICTED_SPEND'))), trunc(max(col('PREDICTED_SPEND')))).toPandas().iloc[0, ]

        st.write('#### 쇼핑몰 고객 연간 소비액 예측')
        met1,ans1,met2,emp1 = st.columns([4,1,4,10])
        with met1:
            st.metric(label="최소", value=f"${int(minspend)}", label_visibility="collapsed")
            st.write("<h5 style='text-align: center; color: #000080; '>최소</h5>", unsafe_allow_html=True)
        with ans1:
            st.write("<h5 style='text-align: center; font-size: 50px; color: #000080;'><strong>~</strong></h5>", unsafe_allow_html=True)
        with met2:
            st.metric(label="최대", value=f"${int(maxspend)}", label_visibility="collapsed")
            st.write("<h5 style='text-align: center; color: #000080; '>최대</h5>", unsafe_allow_html=True)
    except:
        st.write("<h5 style=' font-size: 22px; color: #000080;'><strong>조회 할 수 없는 데이터입니다.</strong></h5>", unsafe_allow_html=True)

empty1,col1 ,empty2= st.columns([3.1, 12, 4])
with col1:    
    st.markdown("----")
    st.markdown("## Data Analysis")

empty1, col1, empty1, col2, empty2= st.columns([3, 6.2, 0.4, 5.6, 3])
with col1:
    st.write("#### 특성 중요도 그래프")
    # 주어진 값들
    SESSION_LENGTH = 0.1
    TIME_ON_APP = 0.21
    TIME_ON_WEBSITE = 0.01
    LENGTH_OF_MEMBERSHIP = 0.68

    # 데이터 프레임 생성
    data = pd.DataFrame({
        'Variable': ['Web 평균 이용 시간', 'App 평균 이용 시간', '매장 평균 이용 시간', '맴버쉽 가입 년 수'],
        'Value': [TIME_ON_WEBSITE, TIME_ON_APP, SESSION_LENGTH, LENGTH_OF_MEMBERSHIP]
    })

    # 막대 그래프 생성
    bar_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Value:Q', title='중요도', scale=alt.Scale(domain=(0, 0.8))),  # x축 변경 및 막대의 범위 축소
        y=alt.Y('Variable:N', sort=None, title=None, axis=alt.Axis(labelFontSize=11, labelPadding=10, labelColor='black')),  # y축 타이틀 제거 및 칸 간격 조정
        color=alt.Color('Variable:N', legend=None),
    ).properties(
        width=300,  # 그래프 너비 설정
        height=200  # 그래프 높이 설정
    )

    # 텍스트 폰트 크기를 조정하는 새로운 막대 그래프 생성
    text_chart = alt.Chart(data).mark_text(
        align='left',
        baseline='middle',
        dx=1,
        fontSize=14,
        color='black',  # 텍스트 색상을 검은색으로 설정
    ).encode(
        x=alt.X('Value:Q'),
        y=alt.Y('Variable:N', sort=None, title=None),
        text=alt.Text('Value:Q', format='.2f'),
    ).properties(
        width=300,  # 그래프 너비 설정
        height=200  # 그래프 높이 설정
    )

    # 그래프 결합
    combined_chart = (bar_chart + text_chart)

    # Streamlit에서 그래프 표시
    st.altair_chart(combined_chart, use_container_width=True)

    quote = "**특성중요도란?**\n\n머신 러닝 모델에서 각 특성이 예측 결과에 얼마나 중요한 역할을 하는지를 평가하는 지표입니다."
    st.info(quote, icon="ℹ️")
with col2:
    st.write("#### 분석 결과")
    st.write("<p style='font-size: 18px;'>쇼핑몰 고객의 활동 특성을 머신러닝으로 분석한 결과,<br> <strong>맴버쉽 가입 년 수</strong>가 고객의 행동에 <strong>가장 큰 영향</strong>을 <br>미친 것으로 나타났습니다. 이러한 결과를 토대로, <br><strong>고객의 맴버쉽 유지를 강화</strong>하는 데 초점을 맞춘<br> 비즈니스 전략을 추진하는 것이 중요합니다.</p>", unsafe_allow_html=True)
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    col1,col2 = st.columns([2,1])
    with col2:
        st.image("dk.png",width=150)

empty1,col1 ,empty2= st.columns([3.1, 12, 4])
with col1:    
    st.markdown("----")
    st.write("## Panas AI🐼")
    matplotlib.use('TkAgg')
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("Your DATA")
        name_define = pd.DataFrame({'컬럼명':['SESSION_LENGTH', 'TIME_ON_APP', 'TIME_ON_WEBSITE', 'LENGTH_OF_MEMBERSHIP', 'PREDICTED_SPEND', 'ACTUAL_SPEND'],
                                    '소비자 행동 특성':['매장 평균 이용 시간 (분)', 'App 평균 이용 시간 (분)', 'Web 평균 이용 시간 (분)', '맴버쉽 가입 년 수','예측 소비액', '실제 소비액']})
        # st.dataframe(name_define)
        st.dataframe(customer_df.toPandas())
    with col2:
        st.info("Chat with your DATA")
        textarea_value = st.text_area("Enter your qurey", "")
        if textarea_value is not None:
            if st.button("Chat with DATA"):
                st.info("Your qurey: "+ str(textarea_value))
                result = my_function(customer_df.toPandas(), str(textarea_value))
                st.success(result)


import streamlit as st
import importlib

# 페이지 설정
st.set_page_config(page_title="가상자산 거래소 통합 대시보드", layout="wide")

# 사이드바 메뉴
st.sidebar.title("분석 선택")
module = st.sidebar.selectbox(
    "분석 항목을 선택하세요",
    (
        "고객 LTV 분석 (LTV)",
        "고객 이탈 예측 (Churn)",
        "전환 분석 (Conversion Funnel)",
    ),
)

# 각 분석 폴더에 있는 .py 파일 불러오기
if module == "고객 LTV 분석 (LTV)":
    ltv = importlib.import_module("Customer_LTV.page_ltv")
    ltv.run()

elif module == "고객 이탈 예측 (Churn)":
    anomaly = importlib.import_module("Customer_Churn_Analysis.page_CCA")
    anomaly.run()

elif module == "전환 분석 (Conversion Funnel)":
    conversion = importlib.import_module("Funnel_Analysis.page_funnel")
    conversion.run()

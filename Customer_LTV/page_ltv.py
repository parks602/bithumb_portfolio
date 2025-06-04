import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go


plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="고객 LTV 분석 리포트")

# 1. 데이터 불러오기
users_df = pd.read_csv("data/users_df.csv", parse_dates=["signup_date"])
transactions_df = pd.read_csv(
    "data/transactions_df.csv", parse_dates=["trade_datetime"]
)
snapshots_df = pd.read_csv("data/snapshots_df.csv", parse_dates=["snapshot_date"])

# 2. LTV 계산
fee_sum = (
    transactions_df.groupby("user_id")["fee_amount"]
    .sum()
    .reset_index()
    .rename(columns={"fee_amount": "total_fee"})
)
avg_assets = (
    snapshots_df.groupby("user_id")["total_asset_value"]
    .mean()
    .reset_index()
    .rename(columns={"total_asset_value": "avg_asset_value"})
)
ltv_segment_df = pd.merge(fee_sum, avg_assets, on="user_id", how="left")
ltv_segment_df = pd.merge(
    ltv_segment_df, users_df[["user_id", "segment"]], on="user_id", how="left"
)
ltv_segment_df["ltv"] = ltv_segment_df["total_fee"]

segment_ltv_stats = (
    ltv_segment_df.groupby("segment")["ltv"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
segment_ltv_stats.columns = ["segment", "mean_ltv", "median_ltv", "user_count"]


# 제목
st.title("📊 고객 LTV 분석 리포트")
st.markdown(
    """
이 리포트는 가상자산 거래소 고객의 Lifetime Value(LTV)를 분석하여 
고객 세그먼트별 가치와 분포를 파악하고, 향후 전략 수립에 활용하기 위한 목적을 가집니다.
"""
)

# ---------------------------------------------------------
# 1. 세그먼트별 통계 요약
# ---------------------------------------------------------
st.subheader(
    f"세그먼트별 LTV 통계 요약({transactions_df['trade_datetime'].min().date()} ~ {transactions_df['trade_datetime'].max().date()})"
)
st.markdown("각 세그먼트의 고객 수, 평균 LTV, 중위값(Median)을 요약한 표입니다.")
st.info(
    "전체 일자를 가지고 통계되었습니다.(= LTV는 개인별 전체 누적 수수료와 같습니다.)"
)
st.dataframe(segment_ltv_stats.round(1))

# ---------------------------------------------------------
# 2. 기초 통계 시각화
# ---------------------------------------------------------
st.subheader("전체 고객의 LTV 기초 통계 시각화")

# 히스토그램 + KDE
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(data=ltv_segment_df, x="ltv", bins=50, kde=True, color="skyblue", ax=ax1)
ax1.set_title("전체 고객의 LTV 분포 (Histogram + KDE)")
ax1.set_xlabel("LTV (KRW)")
ax1.set_ylabel("고객 수")
st.pyplot(fig1)

st.markdown(
    """
- 이 히스토그램은 전체 고객의 LTV 분포를 나타냅니다.
- 대부분 고객의 LTV는 상대적으로 낮은 구간에 몰려 있으며, 일부 고액 고객이 롱테일을 형성하는 **비대칭 분포**를 보입니다.
- KDE 커브는 연속 분포를 추정하여 패턴을 부드럽게 보여줍니다.
"""
)

# Boxplot (세그먼트별)
st.subheader("세그먼트별 LTV Boxplot")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=ltv_segment_df, x="segment", y="ltv", palette="Set2", ax=ax2)
ax2.set_title("Segment별 LTV Boxplot")
ax2.set_xlabel("Segment")
ax2.set_ylabel("LTV (KRW)")
st.pyplot(fig2)

st.markdown(
    """
- 세그먼트별로 LTV 분포를 비교합니다.
- **Large 세그먼트**는 중앙값과 IQR이 현저히 높은 반면, Small 세그먼트는 대부분이 낮은 LTV 구간에 집중되어 있습니다.
- 이상값(outlier)의 유무도 시각적으로 확인할 수 있습니다.
"""
)

# ---------------------------------------------------------
# 3. 세그먼트 필터링
# ---------------------------------------------------------
st.subheader("세그먼트별 유저 상세 확인")

segment_choice = st.selectbox(
    "세그먼트를 선택하세요", ltv_segment_df["segment"].unique()
)
filtered_df = ltv_segment_df[ltv_segment_df["segment"] == segment_choice]
st.markdown(
    f"선택한 세그먼트 **{segment_choice}** 의 유저 상세 정보입니다 (LTV 높은 순 정렬)."
)
st.dataframe(
    filtered_df.round(1).sort_values(by="ltv", ascending=False).reset_index(drop=True)
)


# ---------------------------------------------------------
# 4. 특정 일자 이후 필터링
# ---------------------------------------------------------
st.subheader("일자별 LTV 확인")

date_choice = st.date_input("세그먼트를 선택하세요")
transactions_df["trade_datetime"] = pd.to_datetime(transactions_df["trade_datetime"])

if date_choice:
    date_choice_dt = datetime.combine(date_choice, datetime.min.time())
    selected_fee_sum = (
        transactions_df[transactions_df["trade_datetime"] > date_choice_dt]
        .groupby("user_id")["fee_amount"]
        .sum()
        .reset_index()
        .rename(columns={"fee_amount": "ltv"})
    )

    joined_df = pd.merge(selected_fee_sum, fee_sum, on="user_id", how="left").fillna(0)
    st.markdown(
        f"선택한 날짜 이후의 **{date_choice}** 의 유저 LTV 정보입니다 (LTV 높은 순 정렬)."
    )
    st.dataframe(
        joined_df.round(1).sort_values(by="ltv", ascending=False).reset_index(drop=True)
    )

    graph_choice = joined_df.sort_values(by="ltv", ascending=False).head(30)
    # Plotly 그래프 생성
    fig = go.Figure()

    # 막대그래프 데이터 준비
    users = graph_choice["user_id"]
    ltv_values = graph_choice["ltv"]
    fee_values = graph_choice["total_fee"]  # 수수료 기여도 같이 비교하는 예시

    # Plotly 그래프 구성
    fig = go.Figure(
        data=[
            go.Bar(name="LTV", x=users, y=ltv_values, marker_color="indianred"),
            go.Bar(
                name="Total Fee", x=users, y=fee_values, marker_color="lightskyblue"
            ),
        ]
    )

    fig.update_layout(
        title="LTV 상위 30명 사용자 LTV 및 Total Fee 비교",
        xaxis_title="User ID",
        yaxis_title="값",
        barmode="group",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("LTV와 Total Fee 비교 인사이트 설명", expanded=False):
    st.markdown("####  1. 신규 활동 사용자들의 가치 식별")
    st.markdown(
        """
    - `date_choice` 이후 활동한 사용자 중 최근 거래에서 높은 LTV를 보인 상위 30명을 바로 확인 가능
    - 이들 중 기존 total_fee 대비 LTV가 급증한 사용자는 재활성화되었거나 최근 거래 기여도가 큰 핵심 고객일 가능성 큼
    """
    )

    st.markdown("---")

    st.markdown("####  2. LTV vs. 누적 Total Fee 비교")
    st.markdown(
        """
    - 같은 사용자에 대해 LTV와 Total Fee를 함께 시각화함으로써:

    - **신규 고가치 고객**: LTV는 높지만 Total Fee는 낮은 경우  
        → 최근 유입된 VIP 유저일 수 있음 → 지속적 관심 필요

    - **기존 충성 고객**: 두 지표 모두 높은 경우  
        → VIP 관리, 리텐션 캠페인 타겟

    - **일시적 활동 고객**: LTV는 높지만 누적 기여가 낮은 경우  
        → 단기 이탈 가능성 있음
    """
    )

    st.markdown("---")

    st.markdown("####  3. 날짜 기준 세그먼트 변화 감지")
    st.markdown(
        """
    - 특정 캠페인, 이벤트, UI 변경 이후 유저 LTV가 의미 있게 변했는지 추적 가능
    - 이전과 비교해 어떤 고객군이 더 활성화되었는지, 또는 기대한 반응을 얻지 못했는지 판단
    """
    )

    st.markdown("---")

    st.markdown("####  4. 마케팅/CRM 타겟 선정에 활용")
    st.markdown(
        """
    - LTV 급증 사용자 리스트를 추출해:

    - 이탈 방지 마케팅
    - 추천인 리워드 확대
    - CS 우선 지원 대상 선정
    """
    )

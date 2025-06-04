import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os


def run():

    current_path = os.getcwd()  # 현재 경로
    data_dir = os.path.join(current_path, "data")  # 상위 폴더의 data 폴더 경로

    df = pd.read_csv(f"{data_dir}/funnel_data.csv", parse_dates=True)

    st.title(" 가상자산 거래소 고객 전환 퍼널 분석")

    st.markdown(
        """
    본 대시보드는 **AARRR 프레임워크**에 기반하여, 고객의 가입부터 반복 거래까지의 전환 흐름을 시각화하고 설명합니다.  
    각 단계별 전환율을 파악하고, 이탈이 발생하는 지점을 식별하여 **UX 개선 및 마케팅 전략 수립에 활용**할 수 있습니다.
    """
    )

    # -------------------
    # 전환율 계산
    # -------------------
    step_labels = ["가입", "실명 인증(KYC)", "입금", "첫 거래", "반복 거래"]
    step_cols = [
        "signed_up_at",
        "kyc_at",
        "deposit_at",
        "first_trade_at",
        "repeat_trade_at",
    ]

    conversion_counts = [len(df)]
    for col in step_cols[1:]:
        conversion_counts.append(df[col].notna().sum())

    conversion_rates = [
        (
            f"{(conversion_counts[i+1]/conversion_counts[i])*100:.1f}%"
            if conversion_counts[i] > 0
            else "0%"
        )
        for i in range(len(conversion_counts) - 1)
    ]

    # -------------------
    # 퍼널 차트 시각화
    # -------------------
    st.subheader("🔻 단계별 전환 퍼널")
    col1, col2 = st.columns([2, 1])

    with col1:
        funnel_fig = go.Figure(
            go.Funnel(
                y=step_labels,
                x=conversion_counts,
                textinfo="value+percent previous",
                marker=dict(color="lightskyblue"),
            )
        )
        st.plotly_chart(funnel_fig, use_container_width=True)

    with col2:
        st.markdown("#### 전환율 요약")
        for i in range(len(conversion_rates)):
            if i < len(conversion_rates):
                st.markdown(
                    f"**{step_labels[i]} ➜ {step_labels[i+1]}**: {conversion_rates[i]}"
                )

    # -------------------
    # AARRR 리포트 요약
    # -------------------
    st.subheader(" AARRR 기반 전환 요약")

    st.markdown(
        """
    | 단계 | 설명 | 지표 |
    |------|------|------|
    | **Acquisition** | 유저가 플랫폼에 처음 유입됨 | 가입자 수: **{}명** |
    | **Activation** | 실명인증을 통해 서비스 이용 가능 | 실명 인증 완료율: **{}** |
    | **Retention** | 플랫폼에 재방문할 가능성 있는 단계 | 입금 완료율: **{}** |
    | **Revenue** | 첫 거래를 통해 수익 창출 가능성 확보 | 첫 거래 완료율: **{}** |
    | **Referral** | 만족스러운 경험을 바탕으로 재거래 유도 | 반복 거래 완료율: **{}** |
    """.format(
            conversion_counts[0],
            conversion_rates[0],
            conversion_rates[1],
            conversion_rates[2],
            conversion_rates[3],
        ),
        unsafe_allow_html=True,
    )

    # -------------------
    # 추가: 시간대별 가입자 히트맵 등
    # -------------------
    st.subheader(" 시간 기반 활동 분석")
    df["signed_up_at"] = pd.to_datetime(
        df["signed_up_at"], errors="coerce"
    )  # 문자열 → datetime 변환

    df["signup_hour"] = df["signed_up_at"].dt.hour
    df["signup_date"] = df["signed_up_at"].dt.date

    heatmap_data = df.groupby(["signup_date", "signup_hour"]).size().unstack().fillna(0)

    st.markdown("####  일자 및 시간대별 가입자 분포")
    fig2 = px.imshow(
        heatmap_data,
        labels=dict(x="시간 (시)", y="날짜", color="가입자 수"),
        aspect="auto",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##  최종 결론 및 요약")

    st.markdown(
        f"""
    ###  전환 퍼널 요약 결과
    - **가입 ➜ 실명 인증(KYC)**: **{conversion_rates[0]}**
    - **실명 인증 ➜ 입금**: **{conversion_rates[1]}**
    - **입금 ➜ 첫 거래**: **{conversion_rates[2]}**
    - **첫 거래 ➜ 반복 거래**: **{conversion_rates[3]}**

    ---

    ###  인사이트 요약

    -  **실명 인증까지의 전환율({conversion_rates[0]})이 매우 높음** → 초기 유입 이후의 온보딩은 효과적으로 작동 중입니다.
    -  **입금부터 반복 거래까지 이탈이 점차 심화** → 실제 ‘이용자’로 전환되기까지의 진입 장벽 존재.
    -  **반복 거래 전환율({conversion_rates[3]})은** 비교적 낮은 편 → *첫 거래 이후의 사용자 리텐션 전략*이 필요합니다.

    ---

    ###  개선 방향 제안

    - **첫 거래 후 재거래 유도**:
        - 첫 거래 완료 직후 인앱 메시지, 이메일 리마인더 등으로 행동 강화
        - 추천 리워드 제공 또는 수수료 할인 프로모션 적용

    - **KYC 완료 고객 대상 타깃 프로모션**:
        - 입금 유도를 위한 UI/UX 최적화 또는 프로모션 메시지 제공

    - **전환 흐름 모니터링 자동화**:
        - 유입-이탈 흐름을 실시간 추적할 수 있는 대시보드 및 알림 체계 구축

    ---

    ###  결론

    전환 퍼널 분석을 통해, **고객이 반복 거래 사용자로 전환되기까지의 병목 구간**을 파악할 수 있었습니다.  
    이러한 병목 단계별로 UX 개선 및 리텐션 전략을 수립하면, 고객 생애가치(LTV) 증대와 동시에 **이탈률을 효과적으로 낮출 수 있습니다**.
    """
    )

# 전환 퍼널 분석 대시보드 (Conversion Funnel Analysis for Crypto Exchange)

## 1\. 프로젝트 개요

본 프로젝트는 가상자산 거래소 이용자의 행동 흐름을 분석하여, \*\*회원가입부터 반복 거래까지의 주요 전환율(Funnel)\*\*을 시각화하고 개선 포인트를 도출하기 위한 목적을 가지고 있습니다.
기존 AARRR 프레임워크를 기반으로, 거래소 특유의 사용자 여정을 반영하여 다음과 같은 단계를 추적합니다:

* **가입 (Sign-up)**
* **KYC 인증 완료**
* **첫 입금**
* **첫 거래**
* **반복 거래 (2회 이상)**

각 전환 단계에 도달한 사용자 수, 전환율, 소요 시간 등을 시각화하며, **사용자 세그먼트 및 유입 경로에 따른 차이 분석** 또한 제공합니다.

***

## 2\. 데이터 생성 및 구성

본 프로젝트는 포트폴리오 목적에 따라 **현실적인 사용자 시나리오를 반영한 시뮬레이션 데이터**를 기반으로 합니다.

###  사용자 정보 (1,000명 규모)

* 고객 세그먼트: `light`, `medium`, `heavy`
* 유입 경로: `organic`, `referral`, `paid`
* 가입일: 2024년 1월 1일\~6월 1일 사이 무작위
* 각 사용자에 대해 **시간 기반의 전환 이벤트** 기록 생성

###  주요 이벤트 컬럼

* `signed_up_at`: 가입일시
* `kyc_completed_at`: 실명 인증 완료 일시
* `first_deposit_at`: 첫 입금 시각
* `first_trade_at`: 첫 거래 시각
* `repeated_trade_at`: 2회 이상 거래가 감지된 시점

각 컬럼은 랜덤하게 생성되며, 일부 사용자는 특정 단계에서 이탈하도록 설계되어 **전환율 분석이 가능**합니다.

***

## 3\. 분석 및 대시보드 구성 \(via Streamlit\)

###  전환 퍼널 시각화

* 각 단계별 사용자 수 \& 전환율 표시
* **`st.columns`** 및 **`plotly` 퍼널 차트**로 직관적 표현

###  단계 간 평균 소요 시간

* 각 전환 간 평균 소요 시간 계산
* → 행동 지연이 발생하는 구간 파악 가능

###  사용자 필터링

* 가입일, 유입 경로, 고객군 기준 필터 적용
* → 특정 시점 또는 캠페인 후 유입된 사용자의 퍼널 분석

###  AARRR 리포트 생성

* Acquisition → Activation → Revenue 중심 요약
* 리포트 텍스트는 실시간으로 변화하며 사용자에게 설명 제공

***

## 4\. 향후 확장 방향

* **전환 구간별 UX 개선 포인트 연결**
    (예: KYC 단계 이탈률이 높다면 인증 프로세스 개선 등)
* **실거래 금액 기반 가치 퍼널 분석**
    → 단순 수치보다 수익 기여도를 중심으로 분석
* **이탈 분석과의 통합**
    → 반복 거래 도달 실패자의 행동 분석 및 리타겟팅 전략 수립
* **Sankey Diagram 등 시퀀스 기반 시각화 도입**



"""
============================================================
  마중(MA-JUNG) — 일사량 기반 스마트 버스쉘터 냉방 제어 로직
  팀: 안늦었조 | 이화여자대학교 건축도시시스템공학과
============================================================

[일사량 추정 모델 구조]
  매일 코드 실행 시:
  ① 오늘 SKY 코드(초단기실황) + 오늘 일사량(ASOS 1시간 전) → CSV 누적 저장
  ② 누적 CSV로 선형회귀 모델 학습 (시간대 + 월 + SKY → 일사량)
  ③ 오늘 현재 SKY + 시간대 + 월 → 일사량 예측

  데이터가 쌓일수록 예측 정확도가 향상되는 자기학습형 구조.
  초기(데이터 부족 시)에는 기상청 ASOS 평년값 테이블로 대체.

[전체 냉방 제어 흐름]
  일사량 예측 → H(열부하 지수) → λ(냉각속도계수) → t*(선행시간) → ON/OFF 판단
============================================================
"""

import requests
import math
import os
import csv
from datetime import datetime, timedelta
import pytz

# sklearn은 Colab에 기본 설치되어 있음
from sklearn.linear_model import LinearRegression
import numpy as np

KST = pytz.timezone("Asia/Seoul")


# ============================================================
#  [SECTION 1]  사용자 입력값
# ============================================================

T_실내   = 32.0
T_외기   = 34.0
T_목표   = 26.0
T_기준   = 28.0
ETA_기준 = 15.0

W1 = 0.4
W2 = 0.3
W3 = 0.3

EPSILON = 0.5

API_KEY  = "27cfeb909ef186e6d7fadfdccc6dac5c6d9ffa43133beea2fad0f26da89498fb"
ASOS_STN = "108"
NX, NY   = 55, 127

# Google Drive에 누적 저장할 CSV 경로
# Google Drive 마운트 후 자동으로 여기에 저장됨
CSV_PATH      = "/content/drive/MyDrive/건축도시시스템공학기초설계 - 일사 반영 로직/majung_solar_log.csv"
SKY_TEMP_PATH = "/content/drive/MyDrive/건축도시시스템공학기초설계 - 일사 반영 로직/majung_sky_temp.csv"


# ============================================================
#  [SECTION 2]  Google Drive 마운트
# ============================================================

def mount_google_drive():
    """
    Google Drive를 마운트합니다.
    Colab에서 실행 시 구글 계정 인증 팝업이 뜹니다.
    """
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("  ✓ Google Drive 마운트 완료")
        return True
    except Exception as e:
        print(f"  ✗ Google Drive 마운트 실패: {e}")
        print("  → CSV를 로컬 임시 경로에 저장합니다 (세션 종료 시 삭제됨)")
        return False


# ============================================================
#  [SECTION 3]  오늘 데이터 수집 (SKY + 일사량 쌍)
# ============================================================

def fetch_today_sky(api_key, nx, ny) -> str | None:
    """
    초단기예보(getUltraSrtFcst)에서 현재 시각에 가장 가까운 SKY 코드 수집.
    base_time은 30분 단위 (XX00 또는 XX30).
    현재 시각의 fcstTime과 일치하는 SKY 값을 반환.
    """
    now = datetime.now(KST)

    # base_time: 30분 단위, 현재 분이 30 미만이면 직전 정시, 이상이면 XX30
    if now.minute < 30:
        base = now - timedelta(hours=1)
        base_time = base.strftime("%H") + "30"
    else:
        base_time = now.strftime("%H") + "00"

    base_date = now.strftime("%Y%m%d")
    fcst_time = now.strftime("%H") + "00"   # 현재 시각과 일치하는 예보 시각

    url = (
        f"https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        f"?serviceKey={api_key}&pageNo=1&numOfRows=1000&dataType=JSON"
        f"&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    )

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["response"]["header"]["resultCode"] != "00":
            return None
        items = data["response"]["body"]["items"]["item"]
        if isinstance(items, dict):
            items = [items]
        # 현재 시각 fcstTime과 일치하는 SKY 항목 찾기
        for item in items:
            if item["category"] == "SKY" and item["fcstTime"] == fcst_time:
                return str(item["fcstValue"])
        # 없으면 가장 가까운 SKY 항목 반환
        for item in items:
            if item["category"] == "SKY":
                return str(item["fcstValue"])
        return None
    except:
        return None


def fetch_today_solar(api_key, stn) -> float | None:
    """ASOS에서 1시간 전 실측 일사량 수집 (MJ/m² → W/m²)"""
    now       = datetime.now(KST)
    yesterday = now - timedelta(days=1)
    date_str  = yesterday.strftime("%Y%m%d")
    hour_str  = now.strftime("%H")

    url = (
        f"https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
        f"?serviceKey={api_key}&pageNo=1&numOfRows=1&dataType=JSON"
        f"&dataCd=ASOS&dateCd=HR"
        f"&startDt={date_str}&startHh={hour_str}"
        f"&endDt={date_str}&endHh={hour_str}&stnIds={stn}"
    )

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data["response"]["header"]["resultCode"] != "00":
            return None
        if int(data["response"]["body"]["totalCount"]) == 0:
            return None
        items = data["response"]["body"]["items"]["item"]
        item  = items[0] if isinstance(items, list) else items
        icsr  = item.get("icsr", None)
        if icsr is None or icsr == "":
            return 0.0
        return float(icsr) * 1_000_000 / 3600
    except:
        return None


def save_today_data(csv_path: str, sky: str, i_solar: float):
    """
    오늘 SKY + 일사량 쌍을 CSV에 누적 저장
    컬럼: date, hour, month, sky, i_solar
    """
    now = datetime.now(KST)
    row = {
        "date"   : now.strftime("%Y%m%d"),
        "hour"   : now.hour,
        "month"  : now.month,
        "sky"    : sky,
        "i_solar": round(i_solar, 2),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  ✓ 데이터 저장: {row}")


def load_csv_as_dict(csv_path: str) -> dict:
    """
    CSV를 날짜 기준 딕셔너리로 읽어옴.
    key: date(str), value: 행 데이터(dict)
    """
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["date"]] = row
    return data


def save_csv_from_dict(csv_path: str, data: dict):
    """
    딕셔너리를 날짜 순으로 CSV에 저장.
    """
    if not data:
        return
    fieldnames = ["date", "hour", "month", "sky", "i_solar"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for date in sorted(data.keys()):
            writer.writerow(data[date])


def collect_and_save(api_key, stn, nx, ny, csv_path, sky_temp_path):
    """
    [날짜 기준 행 관리 구조]

    CSV 형식:
    date,     hour, month, sky,  i_solar
    20260611, 11,   6,     ,     819.44   ← 일사량만 (SKY 미수집)
    20260612, 11,   6,     1,           ← SKY만 (일사량은 내일 채워짐)

    실행할 때마다:
    1. 어제 날짜 행에 오늘 수집한 일사량(ASOS 어제 동시간대) 채우기
    2. 오늘 날짜 행에 오늘 SKY 채우기
    → 이틀에 걸쳐 한 행이 완성되는 구조
    """
    print("\n[데이터 수집 및 저장]")
    now       = datetime.now(KST)
    today_str = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    # 기존 CSV 로드
    data = load_csv_as_dict(csv_path)

    # ① 어제 행에 일사량 채우기
    i_solar = fetch_today_solar(api_key, stn)
    if i_solar is not None:
        if yesterday not in data:
            data[yesterday] = {
                "date"   : yesterday,
                "hour"   : now.hour,
                "month"  : now.month,
                "sky"    : "",
                "i_solar": round(i_solar, 2),
            }
        else:
            data[yesterday]["i_solar"] = round(i_solar, 2)
        print(f"  ✓ {yesterday} 일사량 저장: {i_solar:.2f} W/m²")
    else:
        print(f"  ⚠ 일사량 수집 실패")

    # ② 오늘 행에 SKY 채우기
    today_sky = fetch_today_sky(api_key, nx, ny)
    if today_sky is not None:
        if today_str not in data:
            data[today_str] = {
                "date"   : today_str,
                "hour"   : now.hour,
                "month"  : now.month,
                "sky"    : today_sky,
                "i_solar": "",
            }
        else:
            data[today_str]["sky"] = today_sky
        print(f"  ✓ {today_str} SKY 저장: {today_sky}({SKY_LABEL.get(today_sky, '?')})")
    else:
        print(f"  ⚠ SKY 수집 실패")

    # CSV 저장
    save_csv_from_dict(csv_path, data)

    # 완성된 행 (SKY + 일사량 모두 있는 것) 확인
    complete = [(d, r) for d, r in data.items() if r["sky"] != "" and r["i_solar"] != ""]
    print(f"  📊 전체 {len(data)}행 중 완성된 쌍: {len(complete)}개")


# ============================================================
#  [SECTION 4]  누적 데이터로 선형회귀 모델 학습
# ============================================================

SKY_LABEL = {"1": "맑음", "3": "구름많음", "4": "흐림"}

# 평년값 테이블 (최종 fallback)
SOLAR_PEAK_TABLE = {
    0: 0,    1: 0,    2: 0,    3: 0,    4: 0,
    5: 28,   6: 110,  7: 264,  8: 417,  9: 542,
    10: 632, 11: 690, 12: 715, 13: 703, 14: 651,
    15: 557, 16: 430, 17: 287, 18: 148, 19: 55,
    20: 10,  21: 0,   22: 0,   23: 0
}


def load_and_train(csv_path: str):
    """
    누적 CSV를 읽어 선형회귀 모델 학습.
    입력: [hour, month, sky(숫자)]
    출력: i_solar (W/m²)
    데이터 5개 미만이면 학습 불가 → None 반환
    """
    if not os.path.exists(csv_path):
        print("  ⚠ 누적 데이터 없음 — 평년값 테이블 사용")
        return None, 0

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # SKY와 일사량 모두 있는 완성된 행만 학습에 사용
                if row["sky"] == "" or row["i_solar"] == "":
                    continue
                rows.append([
                    int(row["hour"]),
                    int(row["month"]),
                    int(row["sky"]),
                    float(row["i_solar"])
                ])
            except:
                continue

    n = len(rows)
    print(f"  누적 데이터: {n}개")

    if n < 5:
        print(f"  ⚠ 학습 데이터 부족 (최소 5개 필요) — 평년값 테이블 사용")
        return None, n

    X = np.array([[r[0], r[1], r[2]] for r in rows])
    y = np.array([r[3] for r in rows])

    model = LinearRegression()
    model.fit(X, y)

    # 학습 정확도 (R²)
    r2 = model.score(X, y)
    print(f"  ✓ 모델 학습 완료 | R²={r2:.3f} | 데이터 {n}개")
    print(f"  계수: hour={model.coef_[0]:.2f}, month={model.coef_[1]:.2f}, sky={model.coef_[2]:.2f}")

    return model, n


def predict_solar(model, hour: int, month: int, sky: str) -> float:
    """학습된 모델로 일사량 예측"""
    X = np.array([[hour, month, int(sky)]])
    pred = model.predict(X)[0]
    return max(0.0, pred)   # 음수 방지


def get_solar_irradiance(csv_path: str) -> tuple:
    """
    일사량 추정 — 2단계 구조
    1순위: 누적 데이터 선형회귀 모델 예측
    2순위: 평년값 테이블 fallback
    """
    now    = datetime.now(KST)
    hour   = now.hour
    month  = now.month

    # 오늘 SKY 코드
    sky = fetch_today_sky(API_KEY, NX, NY)
    sky_name = SKY_LABEL.get(sky, "알수없음") if sky else "수집실패"

    print(f"\n[일사량 추정]")
    print(f"  현재 SKY={sky}({sky_name}), 시간={hour}시, 월={month}월")

    # 누적 데이터 학습
    print(f"\n  [모델 학습]")
    model, n_data = load_and_train(csv_path)

    if model is not None and sky is not None:
        i_solar = predict_solar(model, hour, month, sky)
        출처 = f"누적학습 선형회귀 모델 ({n_data}개 데이터, SKY={sky}({sky_name}))"
        print(f"  ✓ 예측 일사량: {i_solar:.1f} W/m²")
    else:
        i_solar = float(SOLAR_PEAK_TABLE.get(hour, 0))
        출처 = f"기상청 ASOS 서울 평년값 테이블 ({hour}시 기준)"
        print(f"  → 평년값 사용: {i_solar:.1f} W/m²")

    return i_solar, 출처, sky, sky_name


# ============================================================
#  [SECTION 5~7]  열부하 → λ → t* → 판단 → 절감량
# ============================================================

def calc_heat_load_index(T_indoor, T_outdoor, T_target, T_ref, I_solar) -> float:
    term1 = W1 * (T_indoor  - T_target)
    term2 = W2 * (T_outdoor - T_ref)
    term3 = W3 * (I_solar   / 800.0)
    H = term1 + term2 + term3
    print(f"\n[열부하 지수 H 계산]")
    print(f"  항목1 (실내 온도차): {W1} × ({T_indoor} - {T_target}) = {term1:.3f}")
    print(f"  항목2 (외기 온도차): {W2} × ({T_outdoor} - {T_ref})  = {term2:.3f}")
    print(f"  항목3 (일사량 기여): {W3} × ({I_solar:.1f} / 800)     = {term3:.3f}")
    print(f"  ▶ H = {H:.3f}")
    return H


def determine_lambda(H, I_solar) -> tuple:
    if I_solar > 600 or H > 5.0:
        lam, label = 0.10, "HIGH  (λ=0.10) — 열부하 높음 → 가장 일찍 ON"
    elif I_solar > 200 or H > 2.0:
        lam, label = 0.20, "MID   (λ=0.20) — 열부하 중간, 표준 선행시간"
    else:
        lam, label = 0.35, "LOW   (λ=0.35) — 열부하 낮음 → 늦게 ON 가능"
    print(f"\n[λ 결정]")
    print(f"  I_solar={I_solar:.1f} W/m², H={H:.3f} → {label}")
    return lam, label


def calc_precooling_time(T_start, T_target, epsilon, lam) -> float:
    delta_T = T_start - T_target
    if delta_T <= 0:
        return 0.0
    t_star = (1.0 / lam) * math.log(delta_T / epsilon)
    print(f"\n[냉방 선행시간 t* 계산]")
    print(f"  (1/{lam}) × ln({delta_T}/{epsilon}) = {t_star:.2f}분")
    print(f"  ▶ t* = {t_star:.2f} 분")
    return t_star


def make_control_decision(eta, t_star) -> str:
    print(f"\n[제어 판단]")
    print(f"  ETA={eta:.1f}분  |  t*={t_star:.2f}분")
    if eta <= t_star:
        decision = "🟢 에어컨 ON  — 지금 바로 가동 필요"
    else:
        decision = f"⏳ 대기 중  — {eta - t_star:.1f}분 후 가동 예정"
    print(f"  → {decision}")
    return decision


def calc_energy_saving(t_star, p_ac_kw=2.5) -> dict:
    delta_t_min = max(30.0 - t_star, 0)
    E_saved_kwh = p_ac_kw * (delta_t_min / 60.0)
    E_saved_won = E_saved_kwh * 130
    print(f"\n[에너지 절감량]")
    print(f"  기존 30분 고정 vs 마중 {t_star:.1f}분 → 절감 {delta_t_min:.1f}분")
    print(f"  1회 절감: {E_saved_kwh:.4f} kWh / 약 {E_saved_won:.0f}원")
    print(f"  월간 절감(30회/일): 약 {E_saved_won*30:.0f}원/월")
    return {"delta_t_min": delta_t_min, "E_saved_kwh": E_saved_kwh,
            "E_saved_won": E_saved_won, "E_monthly_won": E_saved_won * 30}


# ============================================================
#  [MAIN]
# ============================================================

def main():
    print("=" * 60)
    print("  마중(MA-JUNG) 냉방 선행시간 산출 시스템")
    print("=" * 60)

    now_kst = datetime.now(KST)
    print(f"\n  현재 시각 (KST): {now_kst.strftime('%Y-%m-%d %H:%M')}")

    # Google Drive 마운트
    print("\n▶ Google Drive 연결")
    drive_ok = mount_google_drive()
    if not drive_ok:
        global CSV_PATH
        CSV_PATH = "/content/majung_solar_log.csv"
        print(f"  → 임시 경로 사용: {CSV_PATH}")

    # 오늘 데이터 수집 & 저장 (다음 실행 때 학습에 사용됨)
    print("\n▶ 오늘 데이터 수집 및 누적 저장")
    collect_and_save(API_KEY, ASOS_STN, NX, NY, CSV_PATH, SKY_TEMP_PATH)

    # 일사량 추정 (누적 모델 or 평년값)
    print("\n▶ 일사량 추정")
    I_solar, 출처, sky, sky_name = get_solar_irradiance(CSV_PATH)

    print(f"\n  ✅ 최종 일사량: {I_solar:.1f} W/m²  |  출처: {출처}")

    # 열부하 → λ → t*
    print("\n▶ 열부하 지수 계산")
    H = calc_heat_load_index(T_실내, T_외기, T_목표, T_기준, I_solar)

    print("\n▶ 냉각속도계수 λ 결정")
    lam, lam_label = determine_lambda(H, I_solar)

    print("\n▶ 냉방 선행시간 t* 계산")
    t_star = calc_precooling_time(T_실내, T_목표, EPSILON, lam)

    print("\n▶ 에어컨 ON/OFF 판단")
    decision = make_control_decision(ETA_기준, t_star)

    print("\n▶ 에너지 절감량")
    saving = calc_energy_saving(t_star)

    print("\n" + "=" * 60)
    print("  [ 최종 결과 요약 ]")
    print("=" * 60)
    print(f"  일사량 데이터 출처  : {출처}")
    print(f"  현재 하늘상태       : SKY={sky}({sky_name})")
    print(f"  추정 일사량         : {I_solar:.1f} W/m²")
    print(f"  실내 온도           : {T_실내}°C  →  목표: {T_목표}°C")
    print(f"  외기 온도           : {T_외기}°C")
    print(f"  열부하 지수 H       : {H:.3f}")
    print(f"  냉각속도계수 λ     : {lam}  ({lam_label.split('—')[0].strip()})")
    print(f"  냉방 선행시간 t*    : {t_star:.2f} 분")
    print(f"  사용자 ETA          : {ETA_기준} 분")
    print(f"  제어 판단           : {decision}")
    print(f"  1회 에너지 절감     : {saving['E_saved_kwh']:.4f} kWh / 약 {saving['E_saved_won']:.0f}원")
    print(f"  월간 절감 추정      : 약 {saving['E_monthly_won']:.0f}원/월")
    print("=" * 60)


if __name__ == "__main__":
    main()

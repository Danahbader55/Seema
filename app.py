import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import math

import numpy as np

# ── إحداثيات المدن ──
CITY_COORDS = {
    "الرياض":         (24.7136, 46.6753),
    "جدة":            (21.4858, 39.1925),
    "الدمام":         (26.4207, 50.0888),
    "مكة المكرمة":    (21.3891, 39.8579),
    "المدينة المنورة":(24.5247, 39.5692),
    "تبوك":           (28.3838, 36.5550),
    "أبها":           (18.2164, 42.5053),
    "بريدة":          (26.3292, 43.9749),
    "حائل":           (27.5114, 41.7208),
    "الخبر":          (26.2172, 50.1971),
}
CITY_AIRPORTS = {
    "الرياض":"RUH","جدة":"JED","الدمام":"DMM","مكة المكرمة":"JED",
    "المدينة المنورة":"MED","تبوك":"TUU","أبها":"AHB",
    "بريدة":"ELQ","حائل":"HAS","الخبر":"DMM",
}
CITY_DISTRICTS = {
    "الرياض":         ["العليا","النخيل","الملقا","الروضة","السليمانية"],
    "جدة":            ["الزهراء","الروضة","النزهة","الصفا","الحمراء"],
    "الدمام":         ["الشاطئ","العزيزية","الفيصلية","النور","الروضة"],
    "مكة المكرمة":    ["العزيزية","النسيم","الزاهر","العوالي","أجياد"],
    "المدينة المنورة":["العوالي","قباء","العزيزية","النخيل","الدفاع"],
    "تبوك":           ["المروج","الأمل","النزهة","الورود","الصناعية"],
    "أبها":           ["المنهل","النخيل","الحزم","الربوة","السلامة"],
    "بريدة":          ["المطار","العزيزية","الشفاء","النهضة","الروضة"],
    "حائل":           ["الصالحية","الغزالة","السلام","النزهة","الحمراء"],
    "الخبر":          ["العقربية","الثقبة","الراكة","الكورنيش","الصفا"],
}
REVIEW_TEXTS = ["السائق ممتاز","خدمة رائعة","في الوقت","سائق محترم","شكراً","وصل سليم","تأخر قليلاً"]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def generate_flight_data(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    rows = []
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 1, 1)
    suspicious_ids = suspicious_ids or set()
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        n_flights = rng.integers(2, 6) if is_susp else rng.integers(0, 3)
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        for _ in range(n_flights):
            arr_city = rng.choice([c for c in cities if c != home_city])
            dep_time = base_date + timedelta(days=int(rng.integers(0,80)), hours=int(rng.integers(5,22)))
            c1, c2   = CITY_COORDS[home_city], CITY_COORDS[arr_city]
            dist_km  = float(haversine_distance(np.array([c1[0]]), np.array([c1[1]]), np.array([c2[0]]), np.array([c2[1]]))[0])
            duration = int(dist_km / 800 * 60 + 45)
            arr_time = dep_time + timedelta(minutes=duration)
            rows.append({"account_id":acc_id,"flight_date":dep_time.strftime("%Y-%m-%d"),"departure_time":dep_time.strftime("%H:%M"),"arrival_time":arr_time.strftime("%H:%M"),"departure_city":home_city,"arrival_city":arr_city,"duration_min":duration,"distance_km":round(dist_km,1),"airport_code":CITY_AIRPORTS.get(home_city,"UNK")})
    import pandas as pd
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["account_id","flight_date","departure_time","arrival_time","departure_city","arrival_city","duration_min","distance_km","airport_code"])

def generate_review_data(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    import pandas as pd
    rng = np.random.default_rng(seed)
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 1, 1, 8, 0)
    suspicious_ids = suspicious_ids or set()
    rows = []
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        base_lat, base_lon = CITY_COORDS[home_city]
        curr_time = base_date
        for i in range(int(rng.integers(8, 20))):
            if is_susp and i > 0 and rng.random() < 0.25:
                other  = rng.choice([c for c in cities if c != home_city])
                r_lat  = CITY_COORDS[other][0] + rng.uniform(-0.05, 0.05)
                r_lon  = CITY_COORDS[other][1] + rng.uniform(-0.05, 0.05)
                gap    = int(rng.integers(3, 12))
                dist   = rng.choice(CITY_DISTRICTS.get(other, ["غير محدد"]))
            else:
                r_lat  = base_lat + rng.uniform(-0.1, 0.1)
                r_lon  = base_lon + rng.uniform(-0.1, 0.1)
                gap    = int(rng.integers(15, 55))
                dist   = rng.choice(CITY_DISTRICTS.get(home_city, ["غير محدد"]))
            curr_time += timedelta(minutes=gap)
            rows.append({"account_id":acc_id,"review_time":curr_time.strftime("%Y-%m-%d %H:%M"),"district":dist,"rating":int(rng.integers(3,6)),"review_text":rng.choice(REVIEW_TEXTS),"lat":round(float(r_lat),5),"lon":round(float(r_lon),5)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def generate_gps_traces(account_ids, city_map=None, suspicious_ids=None, seed=42):
    from datetime import datetime, timedelta
    import pandas as pd
    rng = np.random.default_rng(seed)
    cities = list(CITY_COORDS.keys())
    base_date = datetime(2024, 3, 1, 8, 0)
    suspicious_ids = suspicious_ids or set()
    rows = []
    for acc_id in account_ids:
        is_susp   = acc_id in suspicious_ids
        home_city = (city_map or {}).get(acc_id, rng.choice(cities))
        if home_city not in CITY_COORDS: home_city = "الرياض"
        curr_lat, curr_lon = CITY_COORDS[home_city]
        curr_time = base_date
        for i in range(int(rng.integers(20, 40))):
            if is_susp and i > 0 and rng.random() < 0.08:
                other   = rng.choice([c for c in cities if c != home_city])
                new_lat = CITY_COORDS[other][0] + rng.uniform(-0.02, 0.02)
                new_lon = CITY_COORDS[other][1] + rng.uniform(-0.02, 0.02)
                gap_sec = int(rng.integers(30, 120))
            else:
                new_lat = curr_lat + rng.uniform(-0.01, 0.01)
                new_lon = curr_lon + rng.uniform(-0.01, 0.01)
                gap_sec = int(rng.integers(120, 500))
            curr_time += timedelta(seconds=gap_sec)
            rows.append({"account_id":acc_id,"timestamp":curr_time.strftime("%Y-%m-%d %H:%M:%S"),"lat":round(float(new_lat),6),"lon":round(float(new_lon),6),"accuracy_m":int(rng.integers(3,15))})
            curr_lat, curr_lon = new_lat, new_lon
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# =========================================
# إعداد الصفحة
# =========================================
st.set_page_config(
    page_title="سِيماء — مرصد السلوك الرقمي",
    page_icon="📊",
    layout="wide"
)
st.title("سِيماء — مرصد السلوك الرقمي")
st.caption("مرصد ذكي لكشف التستر الرقمي في اقتصاد المنصات عبر تحليل المصادر غير التقليدية")

# =========================================
# المعايير الرسمية
# =========================================
BENCHMARKS = {
    "suspicious_ratio": 0.14,
    "max_daily_hours":  14,
    "max_continuous":   10,
    "avg_orders":       10,
    "max_devices":      1,
    "source": "GASTAT Q3 2024 + هيئة النقل سبتمبر 2024 + Uber Global Report 2023",
}

# =========================================
# خرائط أسماء الأعمدة
# =========================================
ACCOUNTS_COL_MAP = {
    "رقم الحساب (Account ID)":                              "account_id",
    "حالة الحساب (Status)":                                  "status",
    "الطلبات المقبولة/يوم (Accepted Orders)":                "accepted_orders",
    "الطلبات الملغاة/يوم (Cancelled Orders)":                "cancelled_orders",
    "ساعات العمل اليومية (Daily Hours)":                     "daily_hours",
    "ساعات العمل الأسبوعية (Weekly Hours)":                  "weekly_hours",
    "تسجيل الدخول/الخروج (Logins)":                          "logins",
    "ساعات العمل المتواصلة (Continuous Work)":               "continuous_work_hours",
    "سرعة قبول الطلب (Acceptance Speed - sec)":              "acceptance_speed_sec",
    "سرعة رفض الطلب (Rejection Speed - sec)":                "rejection_speed_sec",
    "نسبة القبول اليومية (Acceptance Rate)":                  "acceptance_rate",
    "نسبة الرفض اليومية (Rejection Rate)":                    "rejection_rate",
    "عدد الأجهزة المسجلة (Device Count)":                    "device_count",
    "المدينة الرئيسية (Main City)":                          "city",
    "عدد المدن/24 ساعة (City Count)":                        "city_count_per_day",
    "متوسط وقت الراحة بين الطلبات (Rest Interval - min)":    "rest_interval_min",
    "إجمالي المسافة المقطوعة/يوم (Daily Distance - km)":     "daily_distance_km",
}

ORDERS_COL_MAP = {
    "معرّف الطلب":        "order_id",
    "رقم الحساب":         "account_id",
    "تاريخ الطلب":        "order_date",
    "وقت البدء":          "start_time",
    "وقت الانتهاء":       "end_time",
    "مدة الطلب (دقيقة)":  "duration_min",
    "مدينة الطلب":        "order_city",
    "حي الاستلام":        "pickup_district",
    "خط عرض الاستلام":    "pickup_lat",
    "خط طول الاستلام":    "pickup_lon",
    "خط عرض التسليم":     "dropoff_lat",
    "خط طول التسليم":     "dropoff_lon",
    "المسافة (كم)":       "distance_km",
    "حالة الطلب":         "status",
}

FINANCIAL_COL_MAP = {
    "رقم الحساب (Account ID)":                   "account_id",
    "عدد مصادر الشحن المختلفة":                   "charge_sources",
    "عدد مرات السحب يومياً":                      "daily_withdrawals",
    "مجموع المبالغ المسحوبة يومياً (ريال)":        "daily_withdrawn_amount",
    "توقيتات السحب":                              "withdrawal_timing",
    "النمو الشهري في الدخل (%)":                  "monthly_income_growth",
    "عدد تحديثات الحساب البنكي":                  "bank_updates",
}

FLIGHT_COL_MAP = {
    "رقم الحساب (Account ID)":             "account_id",
    "تاريخ الرحلة (Flight Date)":           "flight_date",
    "وقت المغادرة (Departure Time)":        "departure_time",
    "وقت الوصول (Arrival Time)":            "arrival_time",
    "مدينة المغادرة (Departure City)":      "departure_city",
    "مدينة الوصول (Arrival City)":          "arrival_city",
    "مدة الرحلة (Duration - min)":          "duration_min",
    "المسافة (Distance - km)":              "distance_km",
    "رمز المطار (Airport Code)":            "airport_code",
}

REVIEW_COL_MAP = {
    "review_id":"review_id", "order_id":"order_id", "account_id":"account_id",
    "تاريخ التقييم (Review Date)":"review_date",
    "وقت التقييم (Review Time)":"review_time",
    "الحي (District)":"district",
    "خط العرض (Lat)":"lat", "خط الطول (Lon)":"lon",
    "التقييم (Rating)":"rating",
    "نص التقييم (Review Text)":"review_text",
}

GPS_COL_MAP = {
    "gps_id":"gps_id","order_id":"order_id","account_id":"account_id",
    "التاريخ (Date)":"date",
    "الوقت (Timestamp)":"timestamp",
    "خط العرض (Lat)":"lat","خط الطول (Lon)":"lon",
    "دقة الإشارة (Accuracy - m)":"accuracy_m",
}

# =========================================
# دوال مساعدة
# =========================================
def normalize_score(series):
    if series.max() == series.min():
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - series.min()) / (series.max() - series.min()) * 100).round(2)

def compute_zscore(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def generate_reason(row):
    reasons = []
    if row.get("daily_hours", 0) > BENCHMARKS["max_daily_hours"]:
        reasons.append(f"ساعات عمل مرتفعة ({row['daily_hours']:.1f} ساعة/يوم)")
    if row.get("continuous_work_hours", 0) > BENCHMARKS["max_continuous"]:
        reasons.append(f"عمل متواصل {row['continuous_work_hours']:.1f} ساعة بلا توقف")
    if row.get("device_count", 1) > BENCHMARKS["max_devices"]:
        reasons.append(f"أجهزة متعددة ({row['device_count']:.0f})")
    if row.get("city_count_per_day", 1) > 1:
        reasons.append(f"نشاط في {row['city_count_per_day']:.0f} مدن/24 ساعة")
    if row.get("rest_interval_min", 30) < 5:
        reasons.append(f"وقت راحة منخفض جداً ({row['rest_interval_min']:.1f} دقيقة)")
    if row.get("accepted_orders", 0) > 30:
        reasons.append(f"طلبات مرتفعة ({row['accepted_orders']:.0f}/يوم)")
    if row.get("charge_sources", 1) > 3:
        reasons.append(f"مصادر شحن متعددة ({row['charge_sources']:.0f})")
    if row.get("withdrawal_timing", "") == "على مدار 24 ساعة (نشاط مستمر)":
        reasons.append("سحوبات على مدار 24 ساعة")
    if row.get("monthly_income_growth", 0) > 50:
        reasons.append(f"نمو مالي غير طبيعي ({row['monthly_income_growth']:.0f}%)")
    if row.get("bank_updates", 0) > 3:
        reasons.append(f"تحديثات بنكية متكررة ({row['bank_updates']:.0f})")
    if row.get("flight_conflict_score", 0) > 0:
        reasons.append("تعارض مع تذاكر الطيران")
    if row.get("review_conflict_score", 0) > 0:
        reasons.append("تنقل مستحيل في التقييمات")
    if row.get("gps_conflict_score", 0) > 0:
        reasons.append("قفزات GPS مستحيلة")
    return " — ".join(reasons) if reasons else "سلوك ضمن النطاق الطبيعي"


def build_internal_scores(df, contamination):
    # إحداثيات من المدينة
    df["city_lat"] = df["city"].map(lambda c: CITY_COORDS.get(c, (24.0, 45.0))[0])
    df["city_lon"] = df["city"].map(lambda c: CITY_COORDS.get(c, (24.0, 45.0))[1])
    rng = np.random.default_rng(42)
    df["lat"] = df["city_lat"] + rng.uniform(-0.08, 0.08, len(df))
    df["lon"] = df["city_lon"] + rng.uniform(-0.08, 0.08, len(df))

    # Z-Score
    for col in ["daily_hours","continuous_work_hours","accepted_orders","device_count","city_count_per_day"]:
        if col in df.columns:
            df[f"z_{col}"] = compute_zscore(df[col]).round(3)

    # Haversine
    df["dist_from_city_center_km"] = haversine_distance(
        df["lat"].values, df["lon"].values,
        df["city_lat"].values, df["city_lon"].values
    ).round(2)
    df["geo_anomaly_score"] = compute_zscore(df["dist_from_city_center_km"]).abs().round(3)

    # Isolation Forest
    feature_cols = [c for c in [
        "daily_hours","weekly_hours","accepted_orders","cancelled_orders",
        "continuous_work_hours","device_count","city_count_per_day",
        "rest_interval_min","daily_distance_km","logins",
        "acceptance_rate","rejection_rate","geo_anomaly_score",
    ] if c in df.columns]

    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"]         = model.fit_predict(df[feature_cols])
    df["model_score_raw"] = -model.decision_function(df[feature_cols])
    df["model_score"]     = normalize_score(df["model_score_raw"])

    # Rule-based — مبني على معايير رسمية
    df["rule_score"] = 0
    if "daily_hours"           in df.columns: df["rule_score"] += (df["daily_hours"]           > 14).astype(int) * 20
    if "continuous_work_hours" in df.columns: df["rule_score"] += (df["continuous_work_hours"] > 10).astype(int) * 20
    if "device_count"          in df.columns: df["rule_score"] += (df["device_count"]          >  1).astype(int) * 20
    if "city_count_per_day"    in df.columns: df["rule_score"] += (df["city_count_per_day"]    >  1).astype(int) * 15
    if "accepted_orders"       in df.columns: df["rule_score"] += (df["accepted_orders"]       > 30).astype(int) * 15
    if "logins"                in df.columns: df["rule_score"] += (df["logins"]                > 15).astype(int) * 10

    # Z composite
    z_available = [f"z_{c}" for c in ["daily_hours","continuous_work_hours","device_count"] if f"z_{c}" in df.columns]
    df["z_score_signal"] = normalize_score(df[z_available].abs().mean(axis=1)) if z_available else 50.0

    df["internal_suspicion_score"] = (
        0.50 * df["model_score"] +
        0.30 * df["rule_score"]  +
        0.20 * df["z_score_signal"]
    ).round(2)

    return df


def add_financial_scores(df, fin_df):
    """يضيف درجة اشتباه مالية لكل حساب."""
    fin = fin_df.copy()
    fin["fin_score"] = 0
    if "charge_sources"        in fin.columns: fin["fin_score"] += (fin["charge_sources"]        >  3).astype(int) * 25
    if "daily_withdrawals"     in fin.columns: fin["fin_score"] += (fin["daily_withdrawals"]     >  5).astype(int) * 20
    if "monthly_income_growth" in fin.columns: fin["fin_score"] += (fin["monthly_income_growth"] > 50).astype(int) * 25
    if "bank_updates"          in fin.columns: fin["fin_score"] += (fin["bank_updates"]          >  3).astype(int) * 15
    if "withdrawal_timing"     in fin.columns:
        fin["fin_score"] += (fin["withdrawal_timing"] == "على مدار 24 ساعة (نشاط مستمر)").astype(int) * 15

    df = df.merge(
        fin[["account_id","fin_score","charge_sources","daily_withdrawals",
             "monthly_income_growth","bank_updates","withdrawal_timing"]],
        on="account_id", how="left"
    )
    df["fin_score"] = df["fin_score"].fillna(0)
    return df


def detect_flight_conflicts_v2(accounts_df, orders_df, flight_df):
    """
    يكشف: حساب نفّذ طلباً في نفس يوم ووقت رحلته الجوية.
    المنطق: order_date == flight_date AND start_time خلال فترة الرحلة.
    """
    if flight_df.empty or orders_df.empty:
        return pd.DataFrame()

    conflicts = []
    for acc_id in accounts_df["account_id"].unique():
        acc_flights = flight_df[flight_df["account_id"] == acc_id]
        acc_orders  = orders_df[(orders_df["account_id"] == acc_id) & (orders_df["status"] == "مكتمل")]
        if acc_flights.empty or acc_orders.empty:
            continue

        for _, flight in acc_flights.iterrows():
            flight_date = str(flight["flight_date"])[:10]
            dep_time    = str(flight["departure_time"])[:5]
            arr_time    = str(flight["arrival_time"])[:5]

            # طلبات في نفس يوم الرحلة
            same_day_orders = acc_orders[acc_orders["order_date"].astype(str).str[:10] == flight_date]
            if same_day_orders.empty:
                continue

            for _, order in same_day_orders.iterrows():
                order_start = str(order["start_time"])[:5]
                # لو وقت الطلب بعد المغادرة → تعارض
                if order_start >= dep_time:
                    score = 60
                    conflicts.append({
                        "account_id":     acc_id,
                        "order_id":       order["order_id"],
                        "conflict_type":  "نشاط أثناء سفر مسجّل",
                        "conflict_score": score,
                        "flight_date":    flight_date,
                        "flight_route":   f"{flight['departure_city']} → {flight['arrival_city']}",
                        "departure_time": dep_time,
                        "order_time":     order_start,
                        "order_city":     order["order_city"],
                        "evidence": (
                            f"طلب {order['order_id']} نُفِّذ في {order['order_city']} "
                            f"الساعة {order_start} بينما الرحلة أقلعت من "
                            f"{flight['departure_city']} الساعة {dep_time} نفس اليوم"
                        ),
                    })

    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def detect_review_conflicts_v2(orders_df, review_df):
    """
    يكشف: موقع التقييم بعيد عن موقع نهاية الطلب بوقت مستحيل.
    المنطق: order_id → dropoff_lat/lon → review_lat/lon + وقت التقييم - وقت الانتهاء.
    """
    if review_df.empty or orders_df.empty:
        return pd.DataFrame()
    # لو ما في order_id → نرجع فارغ
    if "order_id" not in review_df.columns or "order_id" not in orders_df.columns:
        return pd.DataFrame()
    needed = ["order_id","end_time","dropoff_lat","dropoff_lon","order_date"]
    if not all(c in orders_df.columns for c in needed):
        return pd.DataFrame()
    merged = review_df.merge(
        orders_df[needed],
        on="order_id", how="inner"
    )
    if merged.empty:
        return pd.DataFrame()

    conflicts = []
    for _, row in merged.iterrows():
        try:
            rev_time   = pd.to_datetime(f"{row['review_date']} {row['review_time']}", errors="coerce")
            order_end  = pd.to_datetime(f"{row['order_date']} {row['end_time']}", errors="coerce")
            if pd.isna(rev_time) or pd.isna(order_end):
                continue
            gap_min = (rev_time - order_end).total_seconds() / 60
            if gap_min < 0 or gap_min > 120:
                continue
            dist_km = haversine_scalar(
                float(row["dropoff_lat"]), float(row["dropoff_lon"]),
                float(row["lat"]), float(row["lon"])
            )
            if gap_min > 0:
                speed = dist_km / (gap_min / 60)
            else:
                speed = 0

            if speed > 120:
                conflicts.append({
                    "account_id":          row["account_id"],
                    "order_id":            row["order_id"],
                    "review_id":           row["review_id"],
                    "conflict_type":       "تنقل مستحيل بين نهاية الطلب والتقييم",
                    "conflict_score":      min(60, round(speed / 120 * 40, 1)),
                    "gap_min":             round(gap_min, 1),
                    "dist_km":             round(dist_km, 2),
                    "speed_kmh":           round(speed, 1),
                    "evidence": (
                        f"التقييم جاء من موقع يبعد {dist_km:.1f} كم عن موقع التسليم "
                        f"في {gap_min:.0f} دقيقة (سرعة {speed:.0f} كم/ساعة)"
                    ),
                })
        except Exception:
            continue

    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def detect_gps_conflicts_v2(orders_df, gps_df):
    """
    يكشف: أول نقطة GPS للطلب بعيدة عن موقع الاستلام بوقت مستحيل.
    المنطق: order_id → pickup_lat/lon → أول نقطة GPS + الفارق الزمني.
    """
    if gps_df.empty or orders_df.empty:
        return pd.DataFrame()
    # لو ما في order_id → نرجع فارغ
    if "order_id" not in gps_df.columns or "order_id" not in orders_df.columns:
        return pd.DataFrame()
    needed = ["order_id","account_id","start_time","order_date","pickup_lat","pickup_lon"]
    if not all(c in orders_df.columns for c in needed):
        return pd.DataFrame()
    if "timestamp" not in gps_df.columns:
        return pd.DataFrame()

    conflicts = []
    # أول نقطة GPS لكل طلب
    first_gps = gps_df.sort_values("timestamp").groupby("order_id").first().reset_index()
    merged = first_gps.merge(
        orders_df[needed],
        on="order_id", how="inner"
    )

    for _, row in merged.iterrows():
        try:
            gps_time   = pd.to_datetime(f"{row['date']} {row['timestamp']}", errors="coerce")
            order_start = pd.to_datetime(f"{row['order_date']} {row['start_time']}", errors="coerce")
            if pd.isna(gps_time) or pd.isna(order_start):
                continue
            gap_sec = abs((gps_time - order_start).total_seconds())
            if gap_sec > 1800:
                continue
            dist_km = haversine_scalar(
                float(row["pickup_lat"]), float(row["pickup_lon"]),
                float(row["lat"]), float(row["lon"])
            )
            if gap_sec > 0:
                speed = dist_km / (gap_sec / 3600)
            else:
                speed = 0

            if speed > 150:
                conflicts.append({
                    "account_id":    row["account_id"],
                    "order_id":      row["order_id"],
                    "conflict_type": "قفزة جغرافية مستحيلة في GPS",
                    "conflict_score": min(60, round(speed / 150 * 40, 1)),
                    "gap_sec":       round(gap_sec, 0),
                    "dist_km":       round(dist_km, 2),
                    "speed_kmh":     round(speed, 1),
                    "evidence": (
                        f"أول نقطة GPS تبعد {dist_km:.1f} كم عن موقع الاستلام "
                        f"في {gap_sec:.0f} ثانية (سرعة {speed:.0f} كم/ساعة)"
                    ),
                })
        except Exception:
            continue

    return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()


def compute_final_scores(df, flight_conflicts, review_conflicts, gps_conflicts):
    """يدمج كل الدرجات في درجة اشتباه نهائية."""

    # درجات التعارضات الخارجية لكل حساب
    for col, conflicts in [
        ("flight_conflict_score", flight_conflicts),
        ("review_conflict_score", review_conflicts),
        ("gps_conflict_score",    gps_conflicts),
    ]:
        if not conflicts.empty and "conflict_score" in conflicts.columns:
            scores = conflicts.groupby("account_id")["conflict_score"].sum().reset_index()
            scores.columns = ["account_id", col]
            df = df.merge(scores, on="account_id", how="left")
        else:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["external_suspicion_score"] = (
        df["flight_conflict_score"] * 0.35 +
        df["review_conflict_score"] * 0.35 +
        df["gps_conflict_score"]    * 0.30
    ).clip(upper=100).round(2)

    # الدرجة النهائية
    df["suspicion_score"] = (
        0.45 * df["internal_suspicion_score"] +
        0.20 * normalize_score(df["fin_score"]) +
        0.35 * df["external_suspicion_score"]
    ).round(2)

    df["risk_level"] = pd.cut(
        df["suspicion_score"], bins=[-1, 30, 60, 100],
        labels=["منخفض", "متوسط", "مرتفع"]
    )
    df["final_flag"] = np.where(
        (df["anomaly"] == -1) | (df["suspicion_score"] >= 60),
        "مشبوه", "طبيعي"
    )
    df["reason"] = df.apply(generate_reason, axis=1)
    return df


def create_pdf_report(df, total, suspicious, estimated, gap_pct):
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    title_style   = ParagraphStyle("title",   fontSize=16, alignment=TA_CENTER, spaceAfter=12, fontName="Helvetica-Bold")
    heading_style = ParagraphStyle("heading", fontSize=12, spaceAfter=6,        fontName="Helvetica-Bold")
    body_style    = ParagraphStyle("body",    fontSize=10, spaceAfter=4,        fontName="Helvetica")
    small_style   = ParagraphStyle("small",   fontSize=8,  alignment=TA_CENTER, fontName="Helvetica-Oblique")

    story = []
    story.append(Paragraph("Seema - Digital Behavior Observatory Report", title_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Executive Summary", heading_style))
    bench_pct = BENCHMARKS["suspicious_ratio"] * 100
    for label, val in [
        ("Total Accounts",        total),
        ("Suspicious Accounts",   suspicious),
        ("Estimated Real Workers",estimated),
        ("Statistical Gap",       f"{gap_pct}%"),
        ("GASTAT Benchmark",      f"{bench_pct}%"),
    ]:
        story.append(Paragraph(f"{label}: {val}", body_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Detection Methodology", heading_style))
    story.append(Paragraph("Layer 1 - Internal (45%): Isolation Forest + Rule-based + Z-Score.", body_style))
    story.append(Paragraph("Layer 2 - Financial (20%): Wallet sources, withdrawal timing, income growth, bank updates.", body_style))
    story.append(Paragraph("Layer 3 - External (35%): Flight tickets + App reviews + GPS traces.", body_style))
    story.append(Paragraph("Benchmarks: Saudi Transport Authority max 14h/day, GASTAT ~14%, Uber avg 10 orders/day.", body_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Top 10 Suspicious Accounts", heading_style))
    top_df = df[df["final_flag"] == "مشبوه"].sort_values("suspicion_score", ascending=False).head(10)

    table_data = [["Account ID", "City", "Hours", "Devices", "Internal", "Financial", "External", "Final"]]
    for _, row in top_df.iterrows():
        table_data.append([
            str(row.get("account_id", "-")),
            str(row.get("city", "-")),
            f"{row.get('daily_hours', 0):.1f}",
            f"{row.get('device_count', 1):.0f}",
            f"{row.get('internal_suspicion_score', 0):.1f}",
            f"{row.get('fin_score', 0):.1f}",
            f"{row.get('external_suspicion_score', 0):.1f}",
            f"{row.get('suspicion_score', 0):.1f}",
        ])

    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWHEIGHT",     (0,0), (-1,-1), 18),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Generated by Seema POC | GASTAT Hackathon 2025", small_style))

    doc.build(story)
    return buf.getvalue()


# =========================================
# Sidebar
# =========================================
st.sidebar.header("إعدادات النظام")
st.sidebar.markdown("### 📁 رفع البيانات")

uploaded_platform = st.sidebar.file_uploader(
    "📊 ملف المنصة — من Uber/Careem (Excel)",
    type=["xlsx"],
    help="seema_comprehensive_activity_data.xlsx — يحتوي 3 أوراق"
)
uploaded_flights = st.sidebar.file_uploader(
    "✈️ تذاكر الطيران — من GACA (Excel)", type=["xlsx"],
    help="flight_data.xlsx"
)
uploaded_reviews = st.sidebar.file_uploader(
    "📱 تقييمات التطبيقات — من Google Maps (Excel)", type=["xlsx"],
    help="review_data.xlsx"
)
uploaded_gps = st.sidebar.file_uploader(
    "🗺️ مسارات GPS — من Strava/OSM (Excel)", type=["xlsx"],
    help="gps_data.xlsx"
)

st.sidebar.markdown("---")
contamination = 0.10
show_only_suspicious = st.sidebar.checkbox("عرض الحسابات المشبوهة فقط", value=True)
st.sidebar.markdown("---")

n_uploaded = sum(x is not None for x in [uploaded_platform, uploaded_flights, uploaded_reviews, uploaded_gps])
if n_uploaded == 0:
    st.sidebar.warning("⬆️ ارفع ملف المنصة للبدء")
else:
    st.sidebar.success(f"✅ تم رفع {n_uploaded}/4 ملفات")

# =========================================
# قراءة البيانات
# =========================================
if uploaded_platform is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:60vh;text-align:center;">
        <div style="font-size:64px;margin-bottom:20px;">📂</div>
        <h2 style="color:#1E3A8A;margin-bottom:12px;">ارفع ملف المنصة للبدء</h2>
        <p style="color:#6B7280;font-size:16px;">
            ارفع ملف <b>seema_comprehensive_activity_data.xlsx</b> من القائمة الجانبية
        </p>
        <p style="color:#9CA3AF;font-size:14px;margin-top:8px;">
            يمكنك إضافة ملفات الطيران والتقييمات والـ GPS لتحليل أعمق
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ورقة 1: بيانات الحسابات
raw_accounts = pd.read_excel(uploaded_platform, sheet_name="بيانات الحسابات (Accounts)")
accounts_df  = raw_accounts.rename(columns=ACCOUNTS_COL_MAP)

# ورقة 2: سجل الطلبات
raw_orders = pd.read_excel(uploaded_platform, sheet_name="سجل الطلبات (Orders Log)")
orders_df  = raw_orders.rename(columns=ORDERS_COL_MAP)

# ورقة 3: البيانات المالية
raw_fin = pd.read_excel(uploaded_platform, sheet_name="البيانات المالية (Financial)")
fin_df  = raw_fin.rename(columns=FINANCIAL_COL_MAP)

st.sidebar.success(f"✅ تم رفع {n_uploaded}/4 ملفات")
st.sidebar.caption("المصدر: 📂 بيانات مرفوعة")

# =========================================
# قراءة ملفات المصادر الخارجية
# =========================================
account_ids = accounts_df["account_id"].tolist()
city_map    = dict(zip(accounts_df["account_id"], accounts_df["city"]))

if uploaded_flights is not None:
    flight_df = pd.read_excel(uploaded_flights).rename(columns=FLIGHT_COL_MAP)
    flight_df["flight_date"] = flight_df["flight_date"].astype(str).str[:10]
else:
    flight_df = generate_flight_data(account_ids, city_map=city_map)
    flight_df["flight_date"] = flight_df["flight_date"].astype(str).str[:10] if "flight_date" in flight_df.columns else ""

if uploaded_reviews is not None:
    review_df = pd.read_excel(uploaded_reviews).rename(columns=REVIEW_COL_MAP)
else:
    review_df = generate_review_data(account_ids, city_map=city_map)

if uploaded_gps is not None:
    gps_df = pd.read_excel(uploaded_gps).rename(columns=GPS_COL_MAP)
    gps_df["timestamp"] = pd.to_datetime(
        gps_df["date"].astype(str) + " " + gps_df["timestamp"].astype(str), errors="coerce"
    )
else:
    gps_df = generate_gps_traces(account_ids, city_map=city_map)
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"], errors="coerce")

# =========================================
# التحليل
# =========================================
with st.spinner("جاري التحليل..."):
    # 1. الدرجة الداخلية
    df = build_internal_scores(accounts_df.copy(), contamination)

    # 2. الدرجة المالية
    df = add_financial_scores(df, fin_df)

    # 3. كشف التعارضات الخارجية
    flight_conflicts = detect_flight_conflicts_v2(df, orders_df, flight_df) if not orders_df.empty else pd.DataFrame()
    review_conflicts = detect_review_conflicts_v2(orders_df, review_df) if not orders_df.empty else pd.DataFrame()
    gps_conflicts    = detect_gps_conflicts_v2(orders_df, gps_df) if not orders_df.empty else pd.DataFrame()

    # 4. الدرجة النهائية
    df = compute_final_scores(df, flight_conflicts, review_conflicts, gps_conflicts)

# =========================================
# إحصاءات عامة
# =========================================
total_accounts      = len(df)
suspicious_accounts = int((df["final_flag"] == "مشبوه").sum())
estimated_workers   = total_accounts - suspicious_accounts
gap_pct             = round(suspicious_accounts / total_accounts * 100, 2)
official_pct        = BENCHMARKS["suspicious_ratio"] * 100
gap_vs_official     = round(gap_pct - official_pct, 2)
display_df          = df[df["final_flag"] == "مشبوه"].copy() if show_only_suspicious else df.copy()

# =========================================
# Tabs
# =========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 النظرة العامة",
    "🌐 المصادر غير التقليدية",
    "🔍 محرك الكشف",
    "📋 دعم القرار",
    "📄 التقارير",
])

# ──────────────────────────────────────────
# Tab 1: النظرة العامة
# ──────────────────────────────────────────
with tab1:
    st.subheader("النظرة العامة")

    interp = (
        "سِيماء ترصد نسبة اشتباه أعلى من المعدل الرسمي — يُرجَّح وجود تستر رقمي"
        if gap_vs_official > 5 else
        "نسبة الاشتباه تتوافق مع المعدل الرسمي الوطني"
        if abs(gap_vs_official) <= 5 else
        "نسبة اشتباه أقل من المعدل الرسمي — البيانات أقل تشوهاً من المتوسط الوطني"
    )

    st.markdown(f"""
    <div style="background:#EFF6FF;padding:18px;border-radius:12px;border:1px solid #BFDBFE;">
        <h3 style="margin:0;color:#1E40AF;">الفجوة الإحصائية المرصودة: {gap_pct}%</h3>
        <p style="margin-top:8px;color:#374151;">
        المعدل الرسمي (GASTAT Q3 2024): {official_pct}% —
        الفجوة: <b>{gap_vs_official:+.2f}%</b>
        </p>
        <p style="color:#6B7280;margin:0;">{interp}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("إجمالي الحسابات",   total_accounts)
    c2.metric("الحسابات المشبوهة", suspicious_accounts)
    c3.metric("العمالة المقدّرة",   estimated_workers)
    c4.metric("الفجوة المرصودة",    f"{gap_pct}%")
    c5.metric("المعدل الرسمي",      f"{official_pct}%")

    col_a, col_b = st.columns(2)
    with col_a:
        city_counts = df.groupby("city", as_index=False)["account_id"].count().rename(columns={"account_id":"عدد"})
        st.plotly_chart(
            px.bar(city_counts, x="city", y="عدد", title="توزيع الحسابات حسب المدينة",
                   labels={"city":"المدينة"}),
            use_container_width=True
        )
    with col_b:
        rs = df["final_flag"].value_counts().reset_index()
        rs.columns = ["flag","count"]
        st.plotly_chart(
            px.pie(rs, names="flag", values="count", title="طبيعي مقابل مشبوه"),
            use_container_width=True
        )

    st.write("### أهم 10 حسابات عالية الاشتباه")
    top10_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","internal_suspicion_score",
        "external_suspicion_score","suspicion_score","risk_level","reason"
    ] if c in df.columns]
    st.dataframe(df.sort_values("suspicion_score", ascending=False)[top10_cols].head(10), use_container_width=True)


# ──────────────────────────────────────────
# Tab 2: المصادر غير التقليدية
# ──────────────────────────────────────────
with tab2:
    st.subheader("🌐 المصادر غير التقليدية")
    st.markdown("""
    سِيماء تكشف التستر الرقمي عبر **4 مصادر** — البيانات تأتي خام والنظام يحلل ويكتشف التعارضات.
    """)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("💰 المالية المشبوهة",    int((df["fin_score"] > 40).sum()))
    m2.metric("✈️ تعارضات الطيران",     len(flight_conflicts) if not flight_conflicts.empty else 0)
    m3.metric("📱 تعارضات التقييمات",   len(review_conflicts) if not review_conflicts.empty else 0)
    m4.metric("🗺️ شذوذات GPS",           len(gps_conflicts)    if not gps_conflicts.empty    else 0)

    # ── المصدر المالي ──
    st.write("---")
    st.write("### 💰 المصدر 1 — البيانات المالية")
    st.info(
        "**المصدر الواقعي:** مزود الدفع الإلكتروني (STC Pay / مدى)\n\n"
        "**ما يكشفه:** مصادر شحن متعددة + سحوبات على مدار 24 ساعة + نمو مالي غير طبيعي = أشخاص متعددون يستخدمون نفس المحفظة"
    )
    fin_cols = [c for c in ["account_id","charge_sources","daily_withdrawals","monthly_income_growth","bank_updates","withdrawal_timing","fin_score"] if c in df.columns]
    fin_display = df[fin_cols].sort_values("fin_score", ascending=False)
    st.dataframe(fin_display.head(10), use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if "charge_sources" in df.columns:
            st.plotly_chart(
                px.histogram(df, x="charge_sources", color="final_flag", nbins=15,
                             title="توزيع مصادر الشحن", labels={"charge_sources":"عدد المصادر"}),
                use_container_width=True
            )
    with col_f2:
        if "monthly_income_growth" in df.columns:
            st.plotly_chart(
                px.histogram(df, x="monthly_income_growth", color="final_flag", nbins=20,
                             title="توزيع النمو الشهري في الدخل",
                             labels={"monthly_income_growth":"النمو (%)"}),
                use_container_width=True
            )

    # ── تذاكر الطيران ──
    st.write("---")
    st.write("### ✈️ المصدر 2 — تذاكر الطيران")
    st.info(
        "**المصدر الواقعي:** بيانات GACA العامة\n\n"
        "**كيف يكشف التستر:** `account_id` + `order_date` من سجل الطلبات يُقارَن مع `flight_date` من ملف الطيران — لو طلب نُفِّذ بعد المغادرة = تعارض"
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**بيانات الطيران الخام:**")
        st.dataframe(flight_df.head(8), use_container_width=True)
    with col_t2:
        if not flight_conflicts.empty:
            st.write("**التعارضات المكتشفة:**")
            st.dataframe(flight_conflicts[["account_id","order_id","flight_route","departure_time","order_time","evidence"]].head(8), use_container_width=True)
        else:
            st.info("لا توجد تعارضات — ارفع ملف المنصة مع ملف الطيران للكشف الحقيقي")

    # ── تقييمات التطبيقات ──
    st.write("---")
    st.write("### 📱 المصدر 3 — تقييمات التطبيقات")
    st.info(
        "**المصدر الواقعي:** تقييمات Uber / Google Maps\n\n"
        "**كيف يكشف التستر:** `order_id` يربط نهاية الطلب (`dropoff_lat/lon`) بموقع التقييم — لو المسافة مستحيلة في الوقت المتاح = شخص آخر أعطى التقييم"
    )
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.write("**بيانات التقييمات الخام:**")
        st.dataframe(review_df.head(8), use_container_width=True)
    with col_r2:
        if not review_conflicts.empty:
            st.write("**التنقلات المستحيلة:**")
            st.dataframe(review_conflicts[["account_id","order_id","gap_min","dist_km","speed_kmh","evidence"]].head(8), use_container_width=True)
        else:
            st.info("لا توجد تعارضات — ارفع ملف المنصة مع ملف التقييمات للكشف الحقيقي")

    # ── GPS ──
    st.write("---")
    st.write("### 🗺️ المصدر 4 — مسارات GPS")
    st.info(
        "**المصدر الواقعي:** Strava / OpenStreetMap\n\n"
        "**كيف يكشف التستر:** `order_id` يربط أول نقطة GPS بموقع الاستلام (`pickup_lat/lon`) — لو المسافة مستحيلة في الفارق الزمني = شخص آخر بدأ الطلب"
    )
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("**بيانات GPS الخام:**")
        st.dataframe(gps_df.head(8), use_container_width=True)
    with col_g2:
        if not gps_conflicts.empty:
            st.write("**القفزات المستحيلة:**")
            st.dataframe(gps_conflicts[["account_id","order_id","gap_sec","dist_km","speed_kmh","evidence"]].head(8), use_container_width=True)
        else:
            st.info("لا توجد تعارضات — ارفع ملف المنصة مع ملف GPS للكشف الحقيقي")

    # تأثير المصادر على الدرجة النهائية
    st.write("---")
    st.write("### ⚖️ تأثير المصادر الأربعة على درجة الاشتباه النهائية")
    fig_comp = px.scatter(
        df.sample(min(150, len(df))),
        x="internal_suspicion_score", y="suspicion_score",
        color="final_flag", size="fin_score",
        hover_data=["account_id","city","external_suspicion_score"],
        title="الداخلية مقابل النهائية (حجم الدائرة = الدرجة المالية)",
        labels={"internal_suspicion_score":"الداخلية","suspicion_score":"النهائية"}
    )
    fig_comp.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_comp, use_container_width=True)


# ──────────────────────────────────────────
# Tab 3: محرك الكشف
# ──────────────────────────────────────────
with tab3:
    st.subheader("🔍 محرك الكشف والتحليل")

    st.markdown("""
    **منهجية سِيماء — 3 طبقات:**

    **الطبقة الداخلية (45%):** Isolation Forest + Rule-based (معايير رسمية) + Z-Score + Haversine

    **الطبقة المالية (20%):** مصادر الشحن + توقيتات السحب + النمو الشهري + تحديثات البنك

    **الطبقة الخارجية (35%):** تذاكر الطيران + تقييمات التطبيقات + مسارات GPS

    الربط عبر: `account_id` للطيران والمالية — `order_id` للتقييمات والـ GPS
    """)

    col_x, col_y = st.columns(2)
    with col_x:
        st.plotly_chart(
            px.scatter(df, x="daily_hours", y="accepted_orders", color="final_flag",
                       size="suspicion_score",
                       hover_data=["account_id","city","device_count","reason"],
                       title="ساعات العمل مقابل عدد الطلبات"),
            use_container_width=True
        )
    with col_y:
        st.plotly_chart(
            px.histogram(df, x="suspicion_score", color="final_flag",
                         nbins=30, title="توزيع درجات الاشتباه النهائية"),
            use_container_width=True
        )

    # Z-Score
    z_cols_available = [c for c in ["z_daily_hours","z_continuous_work_hours","z_device_count","z_accepted_orders"] if c in df.columns]
    if z_cols_available:
        st.write("### توزيع Z-Score")
        z_df = df[["final_flag"] + z_cols_available].melt(id_vars="final_flag", var_name="المؤشر", value_name="Z-Score")
        st.plotly_chart(
            px.box(z_df, x="المؤشر", y="Z-Score", color="final_flag",
                   title="توزيع Z-Score حسب تصنيف الحساب"),
            use_container_width=True
        )

    # Haversine
    st.write("### الانحراف الجغرافي — Haversine")
    st.plotly_chart(
        px.scatter_mapbox(
            df, lat="lat", lon="lon", color="final_flag",
            size="geo_anomaly_score",
            hover_data=["account_id","city","dist_from_city_center_km"],
            zoom=5, center={"lat":24.5,"lon":44.0},
            mapbox_style="carto-positron",
            title="توزيع الحسابات جغرافياً"
        ),
        use_container_width=True
    )

    # Heatmap
    st.write("### خريطة الارتباط")
    corr_cols = [c for c in [
        "daily_hours","accepted_orders","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","external_suspicion_score","suspicion_score"
    ] if c in df.columns]
    corr = df[corr_cols].corr()
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        text=np.round(corr.values,2), texttemplate="%{text}",
        colorscale="RdYlGn_r"
    ))
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.write("### جدول النتائج الكاملة")
    result_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","flight_conflict_score",
        "review_conflict_score","gps_conflict_score",
        "internal_suspicion_score","external_suspicion_score",
        "suspicion_score","risk_level","final_flag","reason"
    ] if c in display_df.columns]
    st.dataframe(display_df[result_cols].sort_values("suspicion_score", ascending=False), use_container_width=True)


# ──────────────────────────────────────────
# Tab 4: دعم القرار
# ──────────────────────────────────────────
with tab4:
    st.subheader("📋 دعم القرار")

    st.write("### مقارنة الأرقام")
    compare_df = pd.DataFrame({
        "الحالة": ["الرقم الإداري","بعد تنقية سِيماء","المعدل الرسمي (GASTAT)"],
        "القيمة": [
            total_accounts, estimated_workers,
            int(total_accounts * (1 - BENCHMARKS["suspicious_ratio"]))
        ]
    })
    st.plotly_chart(
        px.bar(compare_df, x="الحالة", y="القيمة", text="القيمة", color="الحالة",
               title="مقارنة: الإداري vs سِيماء vs GASTAT"),
        use_container_width=True
    )

    st.write("### المعايير الرسمية المعتمدة")
    st.table(pd.DataFrame({
        "المؤشر":  ["حد ساعات العمل","متوسط الطلبات","نسبة المشبوهين"],
        "القيمة":  ["14 ساعة/يوم","10 طلبات/يوم","~14%"],
        "المصدر":  ["هيئة النقل السعودية 2024","Uber Global Report 2023","GASTAT Q3 2024"],
    }))

    st.write("### القطاعات والمدن الأعلى خطورة")
    city_risk = (
        df.groupby("city").agg(
            total=("account_id","count"),
            suspicious=("final_flag", lambda x: (x=="مشبوه").sum())
        ).reset_index()
    )
    city_risk["risk_pct"] = (city_risk["suspicious"] / city_risk["total"] * 100).round(2)
    city_risk = city_risk.sort_values("risk_pct", ascending=False)
    st.dataframe(city_risk, use_container_width=True)

    top_city = city_risk.iloc[0]["city"] if len(city_risk) > 0 else "—"

    st.warning(f"""
    **توصية 1:** مراجعة النشاط في مدينة **{top_city}** لارتفاع نسبة الاشتباه.
    **توصية 2:** الحسابات ذات مصادر شحن > 3 أو سحوبات على مدار 24 ساعة تستحق تدقيقاً فورياً.
    **توصية 3:** ربط سجل الطلبات بتذاكر الطيران كشف تعارضات مباشرة لا تُرصد بالتحليل السلوكي وحده.
    **توصية 4:** عدم الاكتفاء بالعدّ الإداري — الحساب الواحد قد يمثل أكثر من عامل.
    """)


# ──────────────────────────────────────────
# Tab 5: التقارير
# ──────────────────────────────────────────
with tab5:
    st.subheader("📄 التقارير والمخرجات")

    export_cols = [c for c in [
        "account_id","city","daily_hours","continuous_work_hours","device_count",
        "city_count_per_day","fin_score","flight_conflict_score",
        "review_conflict_score","gps_conflict_score",
        "internal_suspicion_score","external_suspicion_score",
        "suspicion_score","risk_level","final_flag","reason"
    ] if c in df.columns]

    csv_data = df[export_cols].sort_values("suspicion_score", ascending=False).to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 تحميل النتائج CSV", data=csv_data, file_name="seema_results.csv", mime="text/csv")

    if st.button("📄 إنشاء تقرير PDF"):
        pdf_bytes = create_pdf_report(df, total_accounts, suspicious_accounts, estimated_workers, gap_pct)
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="seema_report.pdf">اضغط هنا لتحميل التقرير</a>', unsafe_allow_html=True)
        st.success("تم إنشاء التقرير")

    st.write("### ملخص سريع")
    st.table(pd.DataFrame({
        "المؤشر": [
            "إجمالي الحسابات","الحسابات المشبوهة","العمالة المقدّرة",
            "الفجوة المرصودة","المعدل الرسمي","الفجوة عن المرجع",
            "تعارضات الطيران","تعارضات التقييمات","شذوذات GPS",
        ],
        "القيمة": [
            total_accounts, suspicious_accounts, estimated_workers,
            f"{gap_pct}%", f"{official_pct}%", f"{gap_vs_official:+.2f}%",
            len(flight_conflicts) if not flight_conflicts.empty else 0,
            len(review_conflicts) if not review_conflicts.empty else 0,
            len(gps_conflicts)    if not gps_conflicts.empty    else 0,
        ]
    }))

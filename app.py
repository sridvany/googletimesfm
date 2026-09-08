"""
TimesFM 3.0 ile Saatlik Seans İçi Kapanış Tahmini (Nowcast).

Veri: yfinance (60 dakikalık saatlik kapanış serisi)
Model: google/timesfm-3.0-pytorch (zero-shot, fine-tuning yok)
Çıktı: Seans sonu kapanış tahmini + 10-90 quantile bandı + yön olasılığı

UYARI: Yatırım tavsiyesi değildir. Finansal zaman serileri yüksek volatilite
ve gürültü içerir; yönsel isabet rastlantısallık seviyesine yakın olabilir.
"""

import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

MODEL_ID = "google/timesfm-3.0-pytorch"
QUANTILE_LEVELS = np.arange(0.1, 0.91, 0.1)  # 9 Quantile

PRESETS = {
    "ABD hisse": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
    "Endeks / ETF": ["SPY", "QQQ", "^GSPC", "^IXIC", "^VIX", "XU100.IS"],
    "BIST": ["THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS"],
    "Kripto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Emtia": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "Döviz": ["EURUSD=X", "USDTRY=X", "GBPUSD=X", "USDJPY=X"],
}

st.set_page_config(page_title="TimesFM Saatlik Nowcast", page_icon="⏱️", layout="wide")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="TimesFM 3.0 yükleniyor...")
def load_forecaster(batch_size: int = 32):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    try:
        token = st.secrets.get("HF_TOKEN")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig(
        checkpoint_path=MODEL_ID,
        per_core_batch_size=batch_size,
        device=device,
    )
    return TimesFM3Evaluator(config), device


# --------------------------------------------------------------------------
# Veri ve Seans Yardımcıları
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_and_meta(ticker: str, interval: str = "60m", period: str = "730d"):
    """Saatlik veriyi ve seans metadata'sını tek istekte çeker."""
    tk = yf.Ticker(ticker)
    for _ in range(2):
        try:
            df = tk.history(period=period, interval=interval, auto_adjust=True, actions=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df["Close"].dropna()
                
                try:
                    md = tk.get_history_metadata() or {}
                except Exception:
                    md = {}
                
                tz = md.get("exchangeTimezoneName") or md.get("timezone") or "UTC"
                idx = pd.to_datetime(s.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize("UTC")
                try:
                    idx = idx.tz_convert(tz)
                except Exception:
                    pass
                s.index = idx

                meta = {
                    "name": md.get("longName") or md.get("shortName") or "",
                    "currency": md.get("currency") or "",
                    "exchange": md.get("fullExchangeName") or md.get("exchangeName") or "",
                    "tz": tz,
                }
                return s.astype("float64"), meta
        except Exception:
            continue
    return None, {}


def session_bar_groups(s: pd.Series):
    """Gün -> O güne ait barların indis konumları."""
    dates = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return dates, s.groupby(dates).indices


def typical_session_bars(groups: dict, exclude=None) -> int:
    """Tamamlanmış son seansların bar sayısı medyanı."""
    days = sorted(d for d in groups if d != exclude)
    if not days:
        return 0
    counts = [len(groups[d]) for d in days[-11:-1]] or [len(groups[days[-1]])]
    return int(np.median(counts))


TR_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
TR_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]


def fmt_tr(ts: pd.Timestamp) -> str:
    return f"{ts.day} {TR_MONTHS[ts.month - 1]} {ts.year} {TR_DAYS[ts.dayofweek]}"


# --------------------------------------------------------------------------
# İstatistik / Olasılık
# --------------------------------------------------------------------------
def prob_above(quantiles: np.ndarray, x: float) -> float:
    """9 quantile ile P(Y > x) ampirik aşım olasılığı hesabı."""
    q = np.sort(np.asarray(quantiles, dtype="float64"))
    lv = QUANTILE_LEVELS

    if x <= q[0]:
        denom = q[1] - q[0]
        slope = (lv[1] - lv[0]) / denom if denom > 1e-12 else 0.0
        cdf = lv[0] - slope * (q[0] - x)
    elif x >= q[-1]:
        denom = q[-1] - q[-2]
        slope = (lv[-1] - lv[-2]) / denom if denom > 1e-12 else 0.0
        cdf = lv[-1] + slope * (x - q[-1])
    else:
        cdf = float(np.interp(x, q, lv))

    return float(np.clip(1.0 - cdf, 0.005, 0.995))


def _apply_preset():
    choice = st.session_state.get("preset")
    if choice and choice != "—":
        st.session_state["ticker"] = choice


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------
st.title("⏱️ TimesFM 3.0 — Saatlik Seans İçi Kapanış Tahmini (Nowcast)")

with st.expander("Bu model nasıl çalışır?", expanded=False):
    st.markdown(
        """
Bu uygulama, seans içinde gerçekleşen **saatlik (60m) barları** takip eder ve 
seansın kalan saatlerini TimesFM temel modeli ile simüle ederek **günün kapanışını** tahminler.

* **Nowcast Mantığı:** Günlük kapanışı beklemek yerine bugünün gerçekleşen saatlik hareketini girdi (context) olarak alır.
* **Tahmin Ufku:** Gün içinde geriye kalan bar sayısı ($H = \\text{Tipik Seans Barı} - \\text{Tamamlanan Bar}$).
* **Olasılık:** Kapanışın dünkü kapanışa ve anlık fiyata göre yukarıda bitme olasılığı ($P(Y > X)$).
        """
    )

st.session_state.setdefault("ticker", "AAPL")

with st.sidebar:
    st.header("Sembol Seçimi")
    ticker = st.text_input("yfinance Sembolü", key="ticker").strip()

    with st.expander("Hazır Listeden Seç"):
        group = st.selectbox("Varlık Sınıfı", list(PRESETS), key="preset_group")
        st.selectbox("Sembol", ["—"] + PRESETS[group], key="preset", on_change=_apply_preset)

    st.header("Model Parametreleri")
    nc_ctx = st.slider("Saatlik Context Uzunluğu (Bar)", 128, 1024, 512, step=64)

    st.header("Doğrulama (Backtest)")
    run_bt = st.checkbox("Saatlik Backtest Çalıştır", value=False)
    bt_days = st.slider("Test Edilecek Seans Sayısı", 10, 60, 20, step=5, disabled=not run_bt)

    st.divider()
    go = st.button("Tahmin Et", type="primary", use_container_width=True)

if not go:
    st.info("Soldan bir sembol seçip **Tahmin Et** butonuna tıklayın.")
    st.stop()

# --------------------------------------------------------------------------
# Veri Yükleme ve Seans Bölümleme
# --------------------------------------------------------------------------
if not ticker:
    st.error("Lütfen geçerli bir sembol girin.")
    st.stop()

hourly, meta = fetch_intraday_and_meta(ticker)

if hourly is None or len(hourly) < 150:
    st.error(
        f"'{ticker}' için yeterli saatlik veri alınamadı. "
        "Sembolü kontrol edin veya Yahoo'nun saatlik veri sağladığı bir piyasa seçin."
    )
    st.stop()

h_dates, h_groups = session_bar_groups(hourly)
last_bar_time = hourly.index[-1]
target_date_key = pd.Timestamp(last_bar_time).tz_localize(None).normalize()

n_typ = typical_session_bars(h_groups, exclude=target_date_key)
today_pos = h_groups.get(target_date_key, np.array([], dtype=int))
k_elapsed = len(today_pos)

if n_typ <= 0 or k_elapsed == 0:
    st.error("Bugüne ait saatlik veri grubu oluşturulamadı. Piyasa henüz açılmamış olabilir.")
    st.stop()

# Bir önceki seansın kapanış fiyatı (referans)
all_days_sorted = sorted(list(h_groups.keys()))
current_day_idx = all_days_sorted.index(target_date_key) if target_date_key in all_days_sorted else -1

if current_day_idx > 0:
    prev_day_key = all_days_sorted[current_day_idx - 1]
    prev_close_price = float(hourly.iloc[h_groups[prev_day_key][-1]])
else:
    prev_close_price = float(hourly.iloc[0])

# --------------------------------------------------------------------------
# Model Çıkarımı (Inference)
# --------------------------------------------------------------------------
try:
    forecaster, device = load_forecaster()
except Exception as exc:
    st.error(f"Model yüklenemedi: {exc}")
    st.stop()

horizon_h = max(1, n_typ - k_elapsed)
h_log = np.log(hourly.values)
cut = int(today_pos[-1])
h_ctx = h_log[max(0, cut + 1 - nc_ctx): cut + 1].astype("float32")

with st.spinner(f"Kalan {horizon_h} saatlik bar tahmin ediliyor..."):
    h_out = list(
        forecaster.predict_batch(
            [h_ctx],
            horizon=horizon_h,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )[0]

hq = np.asarray(h_out.quantiles)[horizon_h - 1][:9]
point_pred = float(np.exp(np.asarray(h_out.forecast)[horizon_h - 1]))
cur_price = float(hourly.iloc[-1])

p_up_prev = prob_above(hq, float(np.log(prev_close_price)))
p_up_cur = prob_above(hq, float(np.log(cur_price)))
q10 = float(np.exp(hq[0]))
q50 = float(np.exp(hq[4]))
q90 = float(np.exp(hq[-1]))
direction = "ARTACAK" if p_up_prev >= 0.5 else "AZALACAK"

# --------------------------------------------------------------------------
# Sonuç Sunumu
# --------------------------------------------------------------------------
label = f"{ticker}"
if meta.get("name"):
    label += f" — {meta['name']}"

st.subheader(label)
st.caption(
    f"Borsa: {meta.get('exchange', 'Bilinmiyor')} · Para Birimi: {meta.get('currency', '')} "
    f"· Son Bar Saati: {last_bar_time:%d.%m.%Y %H:%M} ({meta.get('tz')})"
)

if k_elapsed >= n_typ:
    st.info("ℹ️ Bu seansın tipik bar süresi dolmuş veya seans tamamlanmış görünüyor. Model 1 adım sonrası için simülasyon yaptı.")

st.markdown(f"## {fmt_tr(target_date_key)} Seans Sonu Tahmini")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tahmini Yön", direction, f"%{max(p_up_prev, 1 - p_up_prev) * 100:.1f} güven")
m2.metric("Kapanış Tahmini (q50)", f"{q50:,.4f}", f"{(q50 / prev_close_price - 1) * 100:+.2f}% dünkü kapanışa göre")
m3.metric("Anlık Fiyat", f"{cur_price:,.4f}", f"{(cur_price / prev_close_price - 1) * 100:+.2f}% gün içi")
m4.metric("Mevcut Fiyattan Yükselme", f"%{p_up_cur * 100:.1f}", help="P(Kapanış > Anlık Fiyat)")

summary_df = pd.DataFrame(
    {
        "Parametre": [
            "İşlenen Seans İçi Bar",
            "Tahmin Ufku (Kalan Bar)",
            "Dünkü Kapanışa Göre Artış Olasılığı",
            "Mevcut Fiyata Göre Artış Olasılığı",
            "10–90 Olasılık Bandı",
            "Model Context Uzunluğu",
        ],
        "Değer": [
            f"{k_elapsed} / {n_typ} saat",
            f"{horizon_h} bar",
            f"%{p_up_prev * 100:.1f}",
            f"%{p_up_cur * 100:.1f}",
            f"{q10:,.4f} — {q90:,.4f}",
            f"{len(h_ctx)} saatlik bar",
        ],
    }
)
st.table(summary_df)

# --- Grafik ---
chart_bars = min(100, len(hourly))
hist_df = pd.DataFrame(
    {
        "zaman": hourly.index[-chart_bars:],
        "fiyat": hourly.values[-chart_bars:],
    }
)

est_close_time = last_bar_time + pd.Timedelta(hours=horizon_h)
fc_df = pd.DataFrame(
    {
        "zaman": [est_close_time],
        "fiyat": [q50],
        "alt": [q10],
        "ust": [q90],
    }
)

line = alt.Chart(hist_df).mark_line(strokeWidth=1.5).encode(
    x=alt.X("zaman:T", title=None),
    y=alt.Y("fiyat:Q", title=None, scale=alt.Scale(zero=False)),
)
band = alt.Chart(fc_df).mark_rule(strokeWidth=3, color="#f28e2b").encode(
    x="zaman:T", y="alt:Q", y2="ust:Q"
)
pt = alt.Chart(fc_df).mark_point(size=80, filled=True, color="#f28e2b").encode(
    x="zaman:T", y="fiyat:Q"
)

st.altair_chart((line + band + pt).properties(height=320), use_container_width=True)

# --------------------------------------------------------------------------
# Saatlik Backtest Modülü
# --------------------------------------------------------------------------
if run_bt:
    st.divider()
    st.subheader(f"Geçmiş Seanslar Üzerinde Saatlik Backtest (Son {bt_days} Seans)")

    valid_days = [d for d in all_days_sorted if d != target_date_key and len(h_groups[d]) >= k_elapsed]

    if len(valid_days) < 5:
        st.warning("Backtest için yeterli geçmiş seans bulunamadı.")
    else:
        test_days = valid_days[-bt_days:]
        ctx_list, horizons, prev_closes, actual_closes = [], [], [], []

        for d in test_days:
            idxs = h_groups[d]
            cut_idx = int(idxs[min(k_elapsed - 1, len(idxs) - 2)])
            
            ctx_list.append(h_log[max(0, cut_idx + 1 - nc_ctx): cut_idx + 1].astype("float32"))
            horizons.append(len(idxs) - 1 - (idxs.tolist().index(cut_idx)))
            
            d_pos = all_days_sorted.index(d)
            p_day = all_days_sorted[d_pos - 1]
            prev_closes.append(float(hourly.iloc[h_groups[p_day][-1]]))
            actual_closes.append(float(hourly.iloc[idxs[-1]]))

        max_h = max(horizons)
        with st.spinner(f"{len(ctx_list)} seans geçmişi simüle ediliyor..."):
            bt_preds = list(
                forecaster.predict_batch(
                    ctx_list,
                    horizon=max_h,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            )

        bt_rows = []
        for hzn, p_close, act_close, pr in zip(horizons, prev_closes, actual_closes, bt_preds):
            q = np.asarray(pr.quantiles)[hzn - 1][:9]
            pu = prob_above(q, float(np.log(p_close)))
            bt_rows.append(
                {
                    "p_up": pu,
                    "pred_up": pu >= 0.5,
                    "actual_up": act_close > p_close,
                }
            )

        bt_res = pd.DataFrame(bt_rows)
        acc = (bt_res["pred_up"] == bt_res["actual_up"]).mean()
        base = max(bt_res["actual_up"].mean(), 1 - bt_res["actual_up"].mean())
        brier = np.mean((bt_res["p_up"] - bt_res["actual_up"].astype(float)) ** 2)

        b1, b2, b3 = st.columns(3)
        b1.metric("Saatlik Model İsabeti", f"%{acc * 100:.1f}")
        b2.metric("Naif Taban (Çoğunluk Sınıfı)", f"%{base * 100:.1f}", f"{(acc - base) * 100:+.1f} pp")
        b3.metric("Brier Skoru", f"{brier:.4f}", "0.25 = Rastgele Tahmin", delta_color="off")

        st.caption(
            f"Backtest, geçmiş {len(test_days)} seansın bugünkü gibi tam {k_elapsed}. saatindeki "
            "verisi kesilerek kalan saatlerin tahmin edilmesi mantığıyla çalıştırılmıştır."
        )

st.divider()
st.caption(
    "TimesFM 3.0 araştırma ve akademik kullanım amaçlıdır; yatırım tavsiyesi içermez."
)

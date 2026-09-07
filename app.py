"""
TimesFM 3.0 ile bir sonraki islem gunu yon tahmini.

Veri: yfinance (gunluk kapanis, split/temettu duzeltmeli)
Model: google/timesfm-3.0-pytorch (zero-shot, fine-tuning yok)
Cikti: artis olasiligi + 10-90 quantile bandi + yon etiketi

UYARI: Yatirim tavsiyesi degildir. Gunluk fiyat serileri random walk'a
cok yakindir; yonsel isabet %50 civarinda beklenmelidir. Backtest
sekmesi bunu kendi verinizde olcmeniz icindir.
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

MODEL_ID = "google/timesfm-3.0-pytorch"
QUANTILE_LEVELS = np.arange(0.1, 0.91, 0.1)  # TimesFM 3.0 -> 9 quantile

PRESETS = [
    "AAPL", "MSFT", "NVDA", "SPY", "QQQ",
    "BTC-USD", "ETH-USD",
    "XU100.IS", "THYAO.IS", "GARAN.IS",
    "GC=F", "CL=F", "EURUSD=X", "USDTRY=X",
]

st.set_page_config(page_title="TimesFM Yon Tahmini", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="TimesFM 3.0 yukleniyor (ilk calistirmada birkac dakika)...")
def load_forecaster(batch_size: int = 32):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    # Gated repo: HF_TOKEN ortam degiskeni veya Streamlit secrets uzerinden
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
# Veri
# --------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(ticker: str, period: str) -> pd.Series:
    """Gunluk kapanis serisi. yfinance ara sira bos doner, 3 kez dener."""
    last_err = None
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index)
                return s.astype("float64")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(
        f"'{ticker}' icin veri alinamadi. Sembolu kontrol edin "
        f"veya birkac dakika sonra tekrar deneyin. ({last_err})"
    )


# --------------------------------------------------------------------------
# Quantile -> olasilik
# --------------------------------------------------------------------------
def prob_above(quantiles: np.ndarray, x: float) -> float:
    """
    9 quantile'i ampirik CDF gibi kullanip P(Y > x) hesaplar.
    Kuyruklarda en distaki iki quantile'in egimiyle dogrusal ekstrapolasyon.
    """
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


def make_contexts(values: np.ndarray, ctx_len: int, n: int):
    """Son n gun icin kayan pencere: (context, onceki_deger, gercek_deger)."""
    T = len(values)
    out = []
    for i in range(T - n, T):
        ctx = values[max(0, i - ctx_len):i]
        out.append((ctx.astype("float32"), float(values[i - 1]), float(values[i])))
    return out


# --------------------------------------------------------------------------
# Arayuz
# --------------------------------------------------------------------------
st.title("📈 TimesFM 3.0 — Bir sonraki islem gunu yon tahmini")

with st.sidebar:
    st.header("Ayarlar")
    ticker = st.selectbox("Sembol", PRESETS, index=0, accept_new_options=True)
    period = st.select_slider(
        "Gecmis veri",
        options=["1y", "2y", "5y", "10y", "max"],
        value="5y",
    )
    ctx_len = st.slider("Context uzunlugu (gun)", 128, 2048, 512, step=64)
    use_log = st.checkbox("Log fiyat uzerinde tahmin et", value=True)
    st.divider()
    run_bt = st.checkbox("Backtest calistir", value=False)
    bt_days = st.slider("Backtest gun sayisi", 20, 250, 60, step=10, disabled=not run_bt)
    st.divider()
    go = st.button("Tahmin et", type="primary", use_container_width=True)

if not go:
    st.info(
        "Soldan bir sembol secip **Tahmin et**'e basin. "
        "Model ilk calistirmada Hugging Face'ten indirilir."
    )
    st.stop()

# --- veri ---
try:
    prices = fetch_prices(ticker, period)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if len(prices) < 130:
    st.error(f"Yetersiz veri: {len(prices)} gozlem. Daha uzun bir donem secin.")
    st.stop()

raw = prices.values.astype("float64")
series = np.log(raw) if use_log else raw
context = series[-ctx_len:].astype("float32")
last_val = float(series[-1])
last_price = float(raw[-1])

# --- model ---
try:
    forecaster, device = load_forecaster()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Model yuklenemedi. TimesFM 3.0 agirliklari gated bir repoda; "
        "HF_TOKEN tanimli mi ve lisansi kabul ettiniz mi kontrol edin.\n\n"
        f"Hata: {exc}"
    )
    st.stop()

# --- tahmin ---
with st.spinner("Tahmin uretiliyor..."):
    out = list(
        forecaster.predict_batch(
            [context],
            horizon=1,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )[0]

point = float(np.asarray(out.forecast).ravel()[0])
qs = np.asarray(out.quantiles).reshape(-1)[:9]

p_up = prob_above(qs, last_val)
direction = "ARTACAK" if p_up >= 0.5 else "AZALACAK"
confidence = max(p_up, 1 - p_up)


def to_price(v: float) -> float:
    return float(np.exp(v)) if use_log else float(v)


def pct(v: float) -> float:
    return (to_price(v) / last_price - 1.0) * 100.0


q10, q50, q90 = to_price(qs[0]), to_price(qs[4]), to_price(qs[-1])

# --- sonuc ---
st.subheader(f"{ticker} — son kapanis {last_price:,.4f} ({prices.index[-1]:%Y-%m-%d})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Yon", direction, f"{confidence * 100:.1f}% güven")
c2.metric("Artis olasiligi", f"{p_up * 100:.1f}%")
c3.metric("Medyan tahmin (q50)", f"{q50:,.4f}", f"{pct(qs[4]):+.2f}%")
c4.metric("Nokta tahmin", f"{to_price(point):,.4f}", f"{pct(point):+.2f}%")

st.caption(
    f"10–90 bandi: {q10:,.4f} — {q90:,.4f} "
    f"({pct(qs[0]):+.2f}% / {pct(qs[-1]):+.2f}%) · "
    f"cihaz: {device} · context: {len(context)} gun"
)

if confidence < 0.55:
    st.warning(
        "Olasilik %50'ye cok yakin. Bu, modelin yon konusunda pratikte "
        "bilgi tasimadigi anlamina gelir — yon etiketini tek basina kullanmayin."
    )

# --- grafik ---
hist_n = min(120, len(prices))
hist = pd.DataFrame(
    {"tarih": prices.index[-hist_n:], "fiyat": raw[-hist_n:], "tur": "gecmis"}
)
next_date = prices.index[-1] + pd.tseries.offsets.BDay(1)
fc = pd.DataFrame(
    {"tarih": [next_date], "fiyat": [q50], "alt": [q10], "ust": [q90]}
)

line = (
    alt.Chart(hist)
    .mark_line(strokeWidth=1.6)
    .encode(x=alt.X("tarih:T", title=None), y=alt.Y("fiyat:Q", title=None, scale=alt.Scale(zero=False)))
)
band = alt.Chart(fc).mark_rule(strokeWidth=3, color="#f28e2b").encode(
    x="tarih:T", y="alt:Q", y2="ust:Q"
)
pt = alt.Chart(fc).mark_point(size=90, filled=True, color="#f28e2b").encode(
    x="tarih:T", y="fiyat:Q"
)
st.altair_chart((line + band + pt).properties(height=330), use_container_width=True)

qdf = pd.DataFrame(
    {
        "quantile": [f"q{int(l * 100)}" for l in QUANTILE_LEVELS],
        "fiyat": [to_price(v) for v in qs],
        "degisim %": [pct(v) for v in qs],
    }
)
with st.expander("Quantile tablosu"):
    st.dataframe(qdf, hide_index=True, use_container_width=True)

# --- backtest ---
if run_bt:
    st.divider()
    st.subheader(f"Backtest — son {bt_days} islem gunu")

    windows = make_contexts(series, ctx_len, bt_days)
    with st.spinner(f"{bt_days} pencere tek batch'te tahmin ediliyor..."):
        preds = list(
            forecaster.predict_batch(
                [w[0] for w in windows],
                horizon=1,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )

    rows = []
    for (ctx, prev, actual), p in zip(windows, preds):
        pq = np.asarray(p.quantiles).reshape(-1)[:9]
        pu = prob_above(pq, prev)
        rows.append(
            {
                "p_up": pu,
                "tahmin": pu >= 0.5,
                "gercek": actual > prev,
            }
        )
    bt = pd.DataFrame(rows)

    acc = (bt["tahmin"] == bt["gercek"]).mean()
    base = max(bt["gercek"].mean(), 1 - bt["gercek"].mean())
    brier = np.mean((bt["p_up"] - bt["gercek"].astype(float)) ** 2)

    b1, b2, b3 = st.columns(3)
    b1.metric("Yonsel isabet", f"{acc * 100:.1f}%")
    b2.metric("Naif taban (hep ayni yon)", f"{base * 100:.1f}%", f"{(acc - base) * 100:+.1f} pp")
    b3.metric("Brier skoru", f"{brier:.4f}", "0.25 = rastgele", delta_color="off")

    st.caption(
        "Brier skoru 0.25'in altindaysa olasilik tahmini rastgeleden iyidir. "
        f"Ortalama artis olasiligi: {bt['p_up'].mean() * 100:.1f}%, "
        f"gercek artis orani: {bt['gercek'].mean() * 100:.1f}%. "
        "Tek sembol ve kisa pencerede bu farklar buyuk olcude gurultudur."
    )

st.divider()
st.caption(
    "Bu uygulama arastirma ve egitim amaclidir, **yatirim tavsiyesi degildir**. "
    "TimesFM 3.0 agirliklari `timesfm-non-commercial-license-v1.0` altindadir; "
    "ticari veya produksiyon kullanimi izinli degildir."
)

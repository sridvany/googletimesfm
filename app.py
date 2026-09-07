"""
TimesFM 3.0 ile bir sonraki işlem günü yön tahmini.

Veri: yfinance (günlük kapanış, split/temettu duzeltmeli)
Model: google/timesfm-3.0-pytorch (zero-shot, fine-tuning yok)
Cikti: artış olasılığı + 10-90 quantile bandı + yön etiketi

UYARI: Yatirim tavsiyesi değildir. Günlük fiyat serileri random walk'a
çok yakindir; yonsel isabet %50 civarinda beklenmelidir. Backtest
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

PRESETS = {
    "ABD hisse": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
    "Endeks / ETF": ["SPY", "QQQ", "^GSPC", "^IXIC", "^VIX", "XU100.IS"],
    "BIST": ["THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS"],
    "Kripto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Emtia": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "Doviz": ["EURUSD=X", "USDTRY=X", "GBPUSD=X", "USDJPY=X"],
}

st.set_page_config(page_title="TimesFM Yön Tahmini", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="TimesFM 3.0 yükleniyor (ilk çalıştırmada birkaç dakika)...")
def load_forecaster(batch_size: int = 32):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    # Gated repo: HF_TOKEN ortam değişkeni veya Streamlit secrets üzerinden
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
def fetch_prices(ticker: str, period=None, start=None, end=None):
    """
    Gunluk kapanis serisi + Yahoo chart metadata'si.

    yf.download yerine Ticker.history kullaniyoruz: ayni istekte donen
    metadata icinde borsanin o anki seans penceresi (currentTradingPeriod)
    var, boylece acik/kapali tespiti icin ayri bir istek gerekmiyor.

    Doner: (Series, meta_dict)
    """
    kwargs = {"start": start, "end": end} if start else {"period": period}
    last_err = None
    for _ in range(3):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(
                interval="1d",
                auto_adjust=True,
                actions=False,
                **kwargs,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df["Close"].dropna()
                idx = pd.to_datetime(s.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                s.index = idx
                try:
                    md = tk.get_history_metadata() or {}
                except Exception:  # noqa: BLE001
                    md = {}
                # Ham metadata pickle'lanamiyor (st.cache_data onu serilestirir),
                # bu yuzden sadece duz alanlari cikarip donuyoruz.
                return s.astype("float64"), meta_from_history(md)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(
        f"'{ticker}' için veri alınamadı. Sembolü kontrol edin "
        f"veya birkaç dakika sonra tekrar deneyin. ({last_err})"
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, tz: str, interval: str = "60m", period: str = "730d"):
    """
    Saatlik kapanis serisi, borsanin saat diliminde. Yahoo 60m icin ~730 gun
    veriyor. Basarisiz olursa None doner (nowcast paneli sessizce atlanir).
    """
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=True, actions=False
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = df["Close"].dropna()
                idx = pd.to_datetime(s.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize("UTC")
                try:
                    idx = idx.tz_convert(tz or "UTC")
                except Exception:  # noqa: BLE001
                    pass
                s.index = idx
                return s.astype("float64")
        except Exception:  # noqa: BLE001
            continue
    return None


def session_bar_groups(s: pd.Series):
    """Gun -> o gune ait barlarin pozisyonlari."""
    dates = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return dates, s.groupby(dates).indices


def typical_session_bars(groups: dict, exclude=None) -> int:
    """Tamamlanmis son ~10 seansin bar sayisi medyani."""
    days = sorted(d for d in groups if d != exclude)
    if not days:
        return 0
    counts = [len(groups[d]) for d in days[-11:-1]] or [len(groups[days[-1]])]
    return int(np.median(counts))


def meta_from_history(md: dict) -> dict:
    """
    Chart metadata'sindan isim, borsa, para birimi ve seans penceresi.
    Sadece str/int/None doner: st.cache_data sonucu pickle'ladigi icin
    ham metadata (DataFrame, tzinfo vb. icerir) dogrudan donulemez.
    """
    ctp = (md.get("currentTradingPeriod") or {}).get("regular") or {}

    def _int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _str(v):
        return str(v) if v else ""

    return {
        "name": _str(md.get("longName") or md.get("shortName")),
        "type": _str(md.get("instrumentType")),
        "currency": _str(md.get("currency")),
        "exchange": _str(md.get("fullExchangeName") or md.get("exchangeName")),
        "tz": _str(md.get("exchangeTimezoneName") or md.get("timezone")),
        "session_start": _int(ctp.get("start")),
        "session_end": _int(ctp.get("end")),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def asset_info(ticker: str) -> dict:
    """Metadata eksik kalirsa yedek: .info (bulut IP'lerinde sik sik bos doner)."""
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:  # noqa: BLE001
        return {}
    return {
        "name": info.get("longName") or info.get("shortName") or "",
        "type": info.get("quoteType", ""),
        "currency": info.get("currency", ""),
        "exchange": info.get("fullExchangeName") or info.get("exchange", ""),
        "market_state": (info.get("marketState") or "").upper(),
        "tz": info.get("exchangeTimezoneName") or "",
    }


def _apply_preset():
    choice = st.session_state.get("preset")
    if choice and choice != "—":
        st.session_state["ticker"] = choice


TR_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
TR_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]


def fmt_tr(ts: pd.Timestamp) -> str:
    return f"{ts.day} {TR_MONTHS[ts.month - 1]} {ts.year} {TR_DAYS[ts.dayofweek]}"


def next_session_date(index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Tahminin ait oldugu gun. Seride hafta sonu gozlemi varsa (kripto, bazi FX)
    piyasa 7/24 kabul edilir, yoksa bir sonraki is gunu alinir.
    Resmi tatiller hesaba katilmaz.
    """
    last = pd.Timestamp(index[-1])
    recent = index[-60:]
    if (recent.dayofweek >= 5).mean() > 0.10:
        return last + pd.Timedelta(days=1)
    return last + pd.tseries.offsets.BDay(1)


def session_status(index: pd.DatetimeIndex, meta: dict, override: str = "Otomatik"):
    """
    Son barin tamamlanmis bir seans mi yoksa devam eden gunun yarim bari mi
    oldugunu belirler.

    Sirasiyla:
      1. Yahoo chart metadata'sindaki seans penceresi (en guvenilir, tatilleri
         de cozer: tatilde Yahoo bir sonraki seansi dondurur, simdi < start olur)
      2. .info -> marketState (bulut IP'lerinde sik sik bos doner)
      3. Son barin tarihi (acik/kapali ayrimi yapamaz -> "unknown")

    Doner: (durum, hedef_gun, yarim_bar_var_mi, kaynak, teshis)
      durum: "open"    seans devam ediyor
             "pre"     su an islem yok ama BUGUNUN kapanisi henuz gerceklesmedi
                       (acilis oncesi veya ogle arasi) -> nowcast calisabilir
             "closed"  gunun seansi bitti, hedef bir sonraki seans
             "unknown" hicbir kaynak karar veremedi
    """
    tz = meta.get("tz") or "UTC"
    try:
        now_local = pd.Timestamp.now(tz=tz)
    except Exception:  # noqa: BLE001
        now_local = pd.Timestamp.now(tz="UTC")
        tz = "UTC"
    today = now_local.normalize().tz_localize(None)

    last = pd.Timestamp(index[-1]).normalize()
    last_is_today = last == today

    diag = {
        "borsa saati": f"{now_local:%Y-%m-%d %H:%M} ({tz})",
        "son günlük bar": f"{last:%Y-%m-%d}",
        "son bar bugün mü": last_is_today,
        "seans penceresi": "yok",
        "marketState": meta.get("market_state") or "yok",
    }

    if override == "Açık say":
        return "open", (last if last_is_today else today), last_is_today, "elle", diag
    if override == "Kapalı say":
        return "closed", next_session_date(index), False, "elle", diag

    # 1) Seans penceresi
    s_start, s_end = meta.get("session_start"), meta.get("session_end")
    if s_start and s_end:
        try:
            start_ts = pd.to_datetime(s_start, unit="s", utc=True).tz_convert(tz)
            end_ts = pd.to_datetime(s_end, unit="s", utc=True).tz_convert(tz)
            diag["seans penceresi"] = f"{start_ts:%Y-%m-%d %H:%M} → {end_ts:%H:%M}"

            if start_ts <= now_local < end_ts:
                return ("open", start_ts.normalize().tz_localize(None),
                        last_is_today, "seans penceresi", diag)
            if now_local >= end_ts:
                return "closed", next_session_date(index), False, "seans penceresi", diag

            # now < start: acilis oncesi, ogle arasi veya tatil.
            # Pencere BUGUNE aitse gunun kapanisi hala onde -> nowcast calisabilir.
            start_day = start_ts.normalize().tz_localize(None)
            if start_day == today:
                return "pre", today, last_is_today, "seans penceresi", diag
            return "closed", start_day, False, "seans penceresi", diag
        except Exception as exc:  # noqa: BLE001
            diag["seans penceresi"] = f"hata: {exc}"

    # 2) marketState
    state = meta.get("market_state", "")
    if state == "PRE":
        return ("pre", (today if not last_is_today else last),
                last_is_today, "marketState", diag)
    if state == "REGULAR":
        return ("open", (last if last_is_today else today),
                last_is_today, "marketState", diag)
    if state in ("POST", "CLOSED", "POSTPOST", "PREPRE"):
        return "closed", next_session_date(index), False, "marketState", diag

    # 3) Son care
    if last_is_today:
        return "unknown", last, True, "tarih karşılaştırması", diag
    return "closed", next_session_date(index), False, "tarih karşılaştırması", diag


# --------------------------------------------------------------------------
# Quantile -> olasılık
# --------------------------------------------------------------------------
def prob_above(quantiles: np.ndarray, x: float) -> float:
    """
    9 quantile'i ampirik CDF gibi kullanıp P(Y > x) hesaplar.
    Kuyruklarda en dıştaki iki quantile'in eğimiyle doğrusal ekstrapolasyon.
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
    """Son n gün için kayan pencere: (context, onceki_deger, gercek_deger)."""
    T = len(values)
    out = []
    for i in range(T - n, T):
        ctx = values[max(0, i - ctx_len):i]
        out.append((ctx.astype("float32"), float(values[i - 1]), float(values[i])))
    return out


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------
st.title("📈 TimesFM 3.0 — Bir sonraki işlem günü yön tahmini")

with st.expander("Bu uygulama ne yapıyor?", expanded=False):
    st.markdown(
        """
**Veri:** yfinance — günlük kapanış, split ve temettü düzeltmeli.

**Model:** `google/timesfm-3.0-pytorch` — zero-shot, fine-tuning yok.

**Çıktı:** artış olasılığı + 10–90 quantile bandı + yön etiketi.

Seçtiğiniz sembolün son N günlük kapanış serisini modele veriyor, bir sonraki
işlem günü için 1 adımlık tahmin alıyor ve modelin döndürdüğü 9 quantile'ı
ampirik dağılım gibi kullanarak `P(yarınki fiyat > bugünkü kapanış)` hesaplıyor.

> **Uyarı:** Yatırım tavsiyesi değildir. Günlük fiyat serileri random walk'a
> çok yakındır; yönsel isabet %50 civarında beklenmelidir. Backtest bölümü
> bunu kendi verinizde ölçmeniz içindir — modelin "hep aynı yön" demenin
> üstüne çıkıp çıkmadığına oradan bakın.
        """
    )

st.session_state.setdefault("ticker", "AAPL")

with st.sidebar:
    st.header("Sembol")
    ticker = st.text_input(
        "yfinance sembolü",
        key="ticker",
        help=(
            "Herhangi bir yfinance sembolü yazabilirsiniz. Örnekler: "
            "AAPL · THYAO.IS · BTC-USD · GC=F · EURUSD=X · ^GSPC · VOD.L · 7203.T"
        ),
    ).strip()

    st.caption(
        "Sembolü bilmiyorsanız "
        "[Yahoo Finance Lookup](https://finance.yahoo.com/lookup) "
        "sayfasından bulup buraya yapıştırın."
    )

    with st.expander("Hazır listeden seç"):
        group = st.selectbox("Varlık sınıfı", list(PRESETS), key="preset_group")
        st.selectbox(
            "Sembol",
            ["—"] + PRESETS[group],
            key="preset",
            on_change=_apply_preset,
        )

    st.header("Veri")
    custom_range = st.checkbox("Özel tarih aralığı", value=False)
    if custom_range:
        today = pd.Timestamp.today().normalize()
        start_d = st.date_input("Başlangıç", value=(today - pd.DateOffset(years=5)).date())
        end_d = st.date_input("Bitiş", value=today.date())
        period = None
    else:
        period = st.select_slider(
            "Geçmiş veri",
            options=["6mo", "1y", "2y", "5y", "10y", "max"],
            value="5y",
        )
        start_d = end_d = None

    session_override = st.radio(
        "Seans durumu",
        ["Otomatik", "Açık say", "Kapalı say"],
        horizontal=True,
        help=(
            "Otomatik: borsa durumu Yahoo'dan okunur. Yahoo yanıt vermezse "
            "elle seçebilirsiniz."
        ),
    )

    st.header("Model")
    ctx_len = st.slider("Context uzunluğu (gün)", 128, 2048, 512, step=64)
    use_log = st.checkbox("Log fiyat üzerinde tahmin et", value=True)

    st.header("Saatlik nowcast")
    run_nc = st.checkbox(
        "Seans içi saatlik tahmin",
        value=True,
        help=(
            "Saatlik barlarla bugünün kapanışını tahmin eder. Günlük modelin "
            "aksine bugünün gerçekleşen hareketini görür. Sadece borsa açıkken."
        ),
    )
    nc_ctx = st.slider(
        "Saatlik context (bar)", 128, 1024, 512, step=64, disabled=not run_nc
    )
    st.divider()
    run_bt = st.checkbox("Backtest çalıştır", value=False)
    bt_days = st.slider("Backtest gün sayısı", 20, 250, 60, step=10, disabled=not run_bt)
    bt_compare = st.checkbox(
        "Saatlik nowcast'i de backtest et",
        value=False,
        disabled=not (run_bt and run_nc),
        help=(
            "Aynı hedef günlerde günlük model ile saatlik nowcast'i yan yana "
            "ölçer. Her geçmiş seans, bugünkü ile aynı saatten kesilir."
        ),
    )
    st.divider()
    go = st.button("Tahmin et", type="primary", use_container_width=True)

if not go:
    st.info(
        "Soldan bir sembol seçip **Tahmin et**'e basın. "
        "Model ilk çalıştırmada Hugging Face'ten indirilir."
    )
    st.stop()

# --- veri ---
if not ticker:
    st.error("Bir sembol girin.")
    st.stop()

try:
    if custom_range:
        if start_d >= end_d:
            st.error("Başlangıç tarihi bitişten önce olmalı.")
            st.stop()
        prices, meta = fetch_prices(ticker, start=str(start_d), end=str(end_d))
    else:
        prices, meta = fetch_prices(ticker, period=period)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if len(prices) < 130:
    st.error(
        f"Yetersiz veri: {len(prices)} gözlem. Daha uzun bir dönem seçin "
        f"(en az 130 işlem günü gerekli)."
    )
    st.stop()

if len(prices) < ctx_len:
    st.info(
        f"Seride {len(prices)} gözlem var, context {ctx_len} güne ayarlı. "
        f"Model mevcut tüm geçmişi kullanacak."
    )

# --- seans durumu: devam eden gunun yarim bari context'e girmemeli ---
if not meta.get("session_start") or not meta.get("name"):
    # metadata eksik: .info yedegini dene, sadece bos alanlari doldur
    for k, v in asset_info(ticker).items():
        if not meta.get(k):
            meta[k] = v

status, target_date, has_partial, src, diag = session_status(
    prices.index, meta, session_override
)

live_price = None
live_date = None
if has_partial:
    live_price = float(prices.iloc[-1])
    live_date = pd.Timestamp(prices.index[-1])
    prices = prices.iloc[:-1]
    if len(prices) < 130:
        st.error("Yarım bar çıkarıldıktan sonra yetersiz veri kaldı.")
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
        "Model yüklenemedi. TimesFM 3.0 ağırlıkları gated bir repoda; "
        "HF_TOKEN tanımlı mi ve lisansı kabul ettiniz mi kontrol edin.\n\n"
        f"Hata: {exc}"
    )
    st.stop()

# --- tahmin ---
with st.spinner("Tahmin üretiliyor..."):
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

# --- saatlik nowcast hesabi (once hesapla, sonra ozetle birlikte goster) ---
nc = None
nc_note = None
if run_nc and status not in ("open", "pre"):
    nc_note = "Borsa kapalı olduğu için saatlik nowcast çalışmadı — bugünün seansı zaten bitti."
elif run_nc:
    hourly = fetch_intraday(ticker, meta.get("tz", ""))
    if hourly is None or len(hourly) < 200:
        nc_note = (
            "Bu sembol için saatlik veri alınamadı. Yahoo bazı borsalarda "
            "(özellikle BIST'te) intraday veri vermiyor."
        )
    else:
        h_dates, h_groups = session_bar_groups(hourly)
        today_key = pd.Timestamp(target_date).normalize()
        n_typ = typical_session_bars(h_groups, exclude=today_key)
        today_pos = h_groups.get(today_key, np.array([], dtype=int))
        k_elapsed = len(today_pos)

        if n_typ <= 0 or k_elapsed == 0:
            nc_note = (
                "Bugüne ait saatlik bar henüz yok (seans yeni açılmış olabilir). "
                "Nowcast için en az bir tamamlanmış saat gerekiyor."
            )
        else:
            horizon_h = max(1, n_typ - k_elapsed)
            h_log = np.log(hourly.values)
            cut = int(today_pos[-1])
            h_ctx = h_log[max(0, cut + 1 - nc_ctx): cut + 1].astype("float32")

            with st.spinner(f"Saatlik nowcast: kalan {horizon_h} bar..."):
                h_out = list(
                    forecaster.predict_batch(
                        [h_ctx], horizon=horizon_h,
                        return_quantiles=True, use_symmetric_averaging=False,
                    )
                )[0]

            hq = np.asarray(h_out.quantiles)[horizon_h - 1][:9]
            hp = float(np.exp(np.asarray(h_out.forecast)[horizon_h - 1]))
            cur = float(hourly.iloc[-1])
            p = prob_above(hq, float(np.log(last_price)))
            nc = {
                "p_up": p,
                "dir": "ARTACAK" if p >= 0.5 else "AZALACAK",
                "p_from_now": prob_above(hq, float(np.log(cur))),
                "q10": float(np.exp(hq[0])),
                "q50": float(np.exp(hq[4])),
                "q90": float(np.exp(hq[-1])),
                "point": hp,
                "cur": cur,
                "k": k_elapsed,
                "n": n_typ,
                "horizon": horizon_h,
                "ctx": len(h_ctx),
                "first_bar": hourly.index[cut + 1 - len(h_ctx)],
                "last_bar": hourly.index[cut],
            }

# --- sonuc ---
label = f"{ticker}"
if meta.get("name"):
    label += f" — {meta['name']}"
st.subheader(label)
st.caption(
    f"Son tamamlanmış kapanış {last_price:,.4f} {meta.get('currency', '')} "
    f"({fmt_tr(prices.index[-1])}) · {meta.get('exchange', '')} "
    f"· {len(prices)} gözlem"
)

if status == "open":
    st.success(
        f"🟢 **Borsa şu anda açık.** Tahminler **bugünün kapanışı** "
        f"({fmt_tr(target_date)}) içindir."
    )
elif status == "pre":
    st.success(
        f"🟡 **Şu anda işlem yok** (açılış öncesi veya seans arası), ama bugünün "
        f"kapanışı henüz gerçekleşmedi. Tahminler **bugünün kapanışı** "
        f"({fmt_tr(target_date)}) içindir."
    )
elif status == "closed":
    st.error(
        f"🔴 **Borsa kapalı.** Tahminler **bir sonraki kapanış** "
        f"({fmt_tr(target_date)}) içindir."
    )
else:
    st.warning(
        f"🟡 **Borsa durumu belirlenemedi** (Yahoo seans bilgisi döndürmedi). "
        f"Seans devam ediyor varsayıldı, hedef gün {fmt_tr(target_date)}. "
        f"Yanlışsa soldan **Kapalı say**'ı seçin."
    )

with st.expander("🔧 Seans teşhisi — durum yanlışsa buraya bakın"):
    st.write(
        {
            **diag,
            "karar": status,
            "kaynak": src,
            "hedef gün": f"{target_date:%Y-%m-%d}",
            "yarım bar atıldı": has_partial,
            "borsa (meta)": meta.get("exchange") or "yok",
            "saat dilimi (meta)": meta.get("tz") or "yok — UTC varsayıldı",
        }
    )
    st.caption(
        "Karar yanlışsa soldaki **Seans durumu** ile elle geçersiz kılın. "
        "'seans penceresi' satırı Yahoo'nun bildirdiği seans; BIST'te öğle "
        "arasında öğleden sonraki seans görünür, bu normaldir."
    )

# ---- ozet: iki modeli yan yana ----
st.markdown(f"## {fmt_tr(target_date)} kapanışı")

ctx_start_d = prices.index[-len(context)]
ctx_end_d = prices.index[-1]

fields = [
    "Yön",
    "Artış olasılığı",
    "Kapanış tahmini (q50)",
    "10–90 bandı",
    "Veri tipi",
    "Veri aralığı (context)",
    "Bar sayısı",
]
daily_col = [
    ("🔺 " if p_up >= 0.5 else "🔻 ") + direction,
    f"{p_up * 100:.1f}%",
    f"{q50:,.4f}",
    f"{q10:,.4f} — {q90:,.4f}",
    "Günlük kapanış (1d)",
    f"{ctx_start_d:%d.%m.%Y} → {ctx_end_d:%d.%m.%Y}",
    f"{len(context)} bar",
]

summary = {"": fields, "📅 Günlük model": daily_col}
if nc:
    summary["⏱️ Saatlik nowcast"] = [
        ("🔺 " if nc["p_up"] >= 0.5 else "🔻 ") + nc["dir"],
        f"{nc['p_up'] * 100:.1f}%",
        f"{nc['q50']:,.4f}",
        f"{nc['q10']:,.4f} — {nc['q90']:,.4f}",
        "Saatlik kapanış (60m)",
        f"{nc['first_bar']:%d.%m.%Y %H:%M} → {nc['last_bar']:%d.%m.%Y %H:%M}",
        f"{nc['ctx']} bar",
    ]

st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
st.caption(
    "**Artış olasılığı** her iki sütunda da dünkü kapanışa göredir. Aradaki fark "
    "**veri aralığı** satırından geliyor: günlük model dünkü kapanışta duruyor, "
    "saatlik model bugünün gerçekleşen barlarını da görüyor."
)

if nc and nc["dir"] != direction:
    st.warning(
        f"**İki model çelişiyor:** günlük model **{direction}**, saatlik nowcast "
        f"**{nc['dir']}** diyor. Bu bir hata değil, farklı bilgi kümeleri. "
        f"Hangisine güveneceğini ekrana bakarak seçme; backtest'te bu sembolde "
        f"hangisinin daha isabetli olduğuna bak."
    )
elif nc:
    st.info(f"İki model de aynı yönde: **{direction}**.")

if nc_note:
    st.caption(f"⏱️ Saatlik nowcast çalışmadı: {nc_note}")

# ---- detay: gunluk model ----
st.divider()
st.markdown("### 📅 Günlük model — detay")
st.caption(
    f"Günlük kapanış barlarıyla 1 adım ileri tahmin. Context {prices.index[-1]:%d.%m.%Y} "
    f"kapanışında bitiyor; bugünün gün içi hareketini **görmüyor**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Yön", direction, f"{confidence * 100:.1f}% güven")
c2.metric("Artış olasılığı", f"{p_up * 100:.1f}%", help="Dünkü kapanışa göre.")
c3.metric("Medyan tahmin (q50)", f"{q50:,.4f}", f"{pct(qs[4]):+.2f}%")
c4.metric("Nokta tahmin", f"{to_price(point):,.4f}", f"{pct(point):+.2f}%")

if live_price is not None:
    intraday_pct = (live_price / last_price - 1.0) * 100.0
    live_val = float(np.log(live_price)) if use_log else live_price
    d1, d2 = st.columns(2)
    d1.metric("Şu anki fiyat", f"{live_price:,.4f}", f"{intraday_pct:+.2f}% (gün içi)")
    d2.metric(
        "Günlük model → buradan yükselme",
        f"{prob_above(qs, live_val) * 100:.1f}%",
        help="P(kapanış > şu anki fiyat), günlük modelin dağılımına göre.",
    )

st.caption(
    f"10–90 bandı: {q10:,.4f} — {q90:,.4f} "
    f"({pct(qs[0]):+.2f}% / {pct(qs[-1]):+.2f}%) · "
    f"cihaz: {device} · context: {len(context)} günlük bar · "
    f"seans kaynağı: {src}"
)

if confidence < 0.55:
    st.warning(
        "Günlük modelin olasılığı %50'ye çok yakın — yön konusunda pratikte "
        "bilgi taşımıyor. Yön etiketini tek başına kullanmayın."
    )

# ---- detay: saatlik nowcast ----
if nc:
    st.divider()
    st.markdown("### ⏱️ Saatlik nowcast — detay")
    st.caption(
        f"Saatlik barlarla seansın kalan {nc['horizon']} barı tahmin edildi, "
        f"son bar bugünün kapanışı sayıldı. Context bugün {nc['last_bar']:%H:%M} "
        f"barında bitiyor; bugünün gerçekleşen hareketini **görüyor**."
    )

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Yön", nc["dir"], f"{max(nc['p_up'], 1 - nc['p_up']) * 100:.1f}% güven")
    n2.metric("Artış olasılığı", f"{nc['p_up'] * 100:.1f}%", help="Dünkü kapanışa göre.")
    n3.metric("Kapanış tahmini (q50)", f"{nc['q50']:,.4f}",
              f"{(nc['q50'] / last_price - 1) * 100:+.2f}%")
    n4.metric("Buradan yükselme", f"{nc['p_from_now'] * 100:.1f}%",
              help="P(kapanış > şu anki fiyat), saatlik modelin dağılımına göre.")

    st.caption(
        f"Seansın {nc['k']}/{nc['n']} saati geçti · 10–90 bandı: "
        f"{nc['q10']:,.4f} — {nc['q90']:,.4f} · nokta: {nc['point']:,.4f} · "
        f"context: {nc['ctx']} saatlik bar"
    )

# --- grafik ---
hist_n = min(120, len(prices))
hist = pd.DataFrame(
    {"tarih": prices.index[-hist_n:], "fiyat": raw[-hist_n:], "tur": "geçmiş"}
)
fc = pd.DataFrame(
    {"tarih": [target_date], "fiyat": [q50], "alt": [q10], "ust": [q90]}
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
    st.subheader(f"Backtest — son {bt_days} işlem günü")

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
                "pred_up": pu >= 0.5,
                "actual_up": actual > prev,
            }
        )
    bt = pd.DataFrame(rows)

    acc = (bt["pred_up"] == bt["actual_up"]).mean()
    base = max(bt["actual_up"].mean(), 1 - bt["actual_up"].mean())
    brier = np.mean((bt["p_up"] - bt["actual_up"].astype(float)) ** 2)

    b1, b2, b3 = st.columns(3)
    b1.metric("Yönsel isabet", f"{acc * 100:.1f}%")
    b2.metric("Naif taban (hep aynı yön)", f"{base * 100:.1f}%", f"{(acc - base) * 100:+.1f} pp")
    b3.metric("Brier skoru", f"{brier:.4f}", "0.25 = rastgele", delta_color="off")

    st.caption(
        "Brier skoru 0.25'in altındaysa olasılık tahmini rastgeleden iyidir. "
        f"Ortalama artış olasılığı: {bt['p_up'].mean() * 100:.1f}%, "
        f"gerçek artış oranı: {bt['actual_up'].mean() * 100:.1f}%. "
        "Tek sembol ve kısa pencerede bu farklar büyük ölçüde gürültüdür."
    )

    # --- ayni gunlerde saatlik nowcast karsilastirmasi ---
    if bt_compare:
        st.markdown("#### Günlük model vs saatlik nowcast")
        hourly_bt = fetch_intraday(ticker, meta.get("tz", ""))
        if hourly_bt is None or len(hourly_bt) < 500:
            st.info("Saatlik veri yetersiz; karşılaştırma yapılamadı.")
        else:
            _, g = session_bar_groups(hourly_bt)
            # bugunku seansta kac bar gectiyse gecmis seanslari da ayni yerden kes
            today_key = pd.Timestamp(target_date).normalize()
            k_cut = len(g.get(today_key, [])) or max(1, typical_session_bars(g) // 2)
            h_log_bt = np.log(hourly_bt.values)

            daily_dates = pd.DatetimeIndex(prices.index).normalize()
            targets = [d for d in daily_dates[-bt_days:] if d in g and len(g[d]) > k_cut]

            if len(targets) < 10:
                st.info(
                    f"Saatlik veriyle örtüşen yalnızca {len(targets)} gün var "
                    f"(Yahoo 60m için ~730 gün veriyor). Karşılaştırma atlandı."
                )
            else:
                ctxs, horizons, refs = [], [], []
                for d in targets:
                    idxs = g[d]
                    cut = int(idxs[k_cut - 1])
                    ctxs.append(
                        h_log_bt[max(0, cut + 1 - nc_ctx): cut + 1].astype("float32")
                    )
                    horizons.append(len(idxs) - k_cut)
                    pos = daily_dates.get_loc(d)
                    refs.append((float(raw[pos - 1]), float(raw[pos])))

                H = max(horizons)
                with st.spinner(f"{len(ctxs)} seans için saatlik nowcast..."):
                    h_preds = list(
                        forecaster.predict_batch(
                            ctxs, horizon=H,
                            return_quantiles=True,
                            use_symmetric_averaging=False,
                        )
                    )

                nrows = []
                for hzn, (prev_c, act_c), pr in zip(horizons, refs, h_preds):
                    q = np.asarray(pr.quantiles)[hzn - 1][:9]
                    pu = prob_above(q, float(np.log(prev_c)))
                    nrows.append({"p_up": pu, "pred_up": pu >= 0.5,
                                  "actual_up": act_c > prev_c})
                nbt = pd.DataFrame(nrows)

                n_acc = (nbt["pred_up"] == nbt["actual_up"]).mean()
                n_brier = np.mean((nbt["p_up"] - nbt["actual_up"].astype(float)) ** 2)
                n_base = max(nbt["actual_up"].mean(), 1 - nbt["actual_up"].mean())

                cmp_df = pd.DataFrame(
                    {
                        "": ["Günlük model", "Saatlik nowcast", "Naif taban"],
                        "Yönsel isabet": [f"{acc * 100:.1f}%", f"{n_acc * 100:.1f}%",
                                          f"{n_base * 100:.1f}%"],
                        "Brier": [f"{brier:.4f}", f"{n_brier:.4f}", "0.2500"],
                        "Gün sayısı": [len(bt), len(nbt), len(nbt)],
                    }
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)
                st.caption(
                    f"Her geçmiş seans, bugünkü ile aynı noktadan kesildi "
                    f"({k_cut}. saatin sonunda) ve kalan barlar tahmin edildi. "
                    f"İki satır farklı gün sayısı içerebilir: saatlik veri {len(nbt)} "
                    f"günü kapsıyor, günlük backtest {len(bt)} günü. "
                    f"Aradaki fark 3–4 puandan küçükse bu örneklem büyüklüğünde "
                    f"anlamlı değildir."
                )

st.divider()
st.caption(
    "Bu uygulama araştırma ve eğitim amaçlıdır, **yatırım tavsiyesi değildir**. "
    "TimesFM 3.0 ağırlıkları `timesfm-non-commercial-license-v1.0` altındadır; "
    "ticari veya prodüksiyon kullanımı izinli değildir."
)

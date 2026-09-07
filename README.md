---
title: TimesFM Yon Tahmini
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: other
---

# TimesFM 3.0 — Bir sonraki işlem günü yön tahmini

yfinance'ten çekilen günlük kapanış serisine Google TimesFM 3.0'ı zero-shot
uygulayıp bir sonraki işlem gününün yönünü, artış olasılığını ve 10–90
quantile bandını gösteren Streamlit uygulaması.

> Yatırım tavsiyesi değildir. Aşağıdaki "Beklentiler" bölümünü okuyun.

## Ne yapıyor

- Sembol seçimi (hisse, endeks, kripto, emtia, döviz — yfinance formatı)
- Log fiyat veya ham fiyat üzerinde tahmin
- 9 quantile'i ampirik CDF gibi kullanarak `P(P_{t+1} > P_t)` hesabı
- Opsiyonel backtest: son N gün için yönsel isabet, naif taban ve Brier skoru

## Beklentiler — önce bunu okuyun

Günlük hisse fiyatı random walk'a çok yakındır. Zero-shot bir foundation
model bu seride pratikte "son değeri devam ettir" davranışına yakınsar ve
yönsel isabet %50 civarında çıkar. Uygulama bu yüzden çıplak bir
ARTACAK/AZALACAK etiketiyle yetinmez; olasılığı ve belirsizlik bandını da
gösterir, olasılık %55'in altındaysa uyarı verir.

Backtest sekmesindeki **naif taban** karşılaştırması önemli: yükseliş
eğilimli bir dönemde "her gün artacak" demek de %53–55 isabet verir. Modelin
değeri, bu tabanı geçip geçmediğinde ve Brier skorunun 0.25'in altında
kalıp kalmadığındadır.

## Lisans uyarısı

TimesFM kaynak kodu Apache-2.0, ancak **3.0 ağırlıkları
`timesfm-non-commercial-license-v1.0`** altında dağıtılıyor ve ticari /
prodüksiyon kullanımı yasak. Bu depo araştırma-eğitim amaçlıdır.
Ticari kullanım gerekiyorsa `app.py` içindeki `MODEL_ID`'yi Apache-2.0 olan
TimesFM 2.5 ağırlıklarıyla değiştirin (API farklı, `timesfm` paketinin 2.x
arayüzü gerekir).

## Kurulum (yerel)

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx          # 3.0 ağırlıkları gated
streamlit run app.py
```

`google/timesfm-3.0-pytorch` sayfasında lisansı kabul etmeniz gerekiyor,
aksi halde indirme 403 döner.

## Dağıtım

### Hugging Face Spaces (önerilen)

330M parametre fp32 ≈ 1.3 GB. **Streamlit Community Cloud'un ücretsiz
belleği bunun için yeterli değil** (yaygın olarak ~1 GB olarak biliniyor);
uygulama açılışta çöker. HF Spaces ücretsiz CPU tier'ı 16 GB RAM verir ve
Streamlit SDK'sını doğrudan destekler.

1. Yeni Space aç → SDK: Streamlit
2. `app.py`, `requirements.txt`, `README.md` dosyalarını push et
   (bu README'nin üstündeki YAML bloğu Spaces için gerekli)
3. Settings → Variables and secrets → `HF_TOKEN` ekle

İlk açılışta model indirilir; sonraki başlatmalar Spaces cache'inden
hızlıdır. `@st.cache_resource` sayesinde model süreç başına bir kez yüklenir.

### Streamlit Community Cloud

Sadece TimesFM 2.5'e geçerseniz mantıklı. Eğitim kurumundaysanız Streamlit'in
kaynak artırım formuyla limit yükseltmesi talep edebilirsiniz.

## Bilinen sorunlar

- **yfinance boş veri / rate limit**: Yahoo zaman zaman bulut IP'lerini
  kısıtlar. `fetch_prices` 3 kez dener ve sonucu 15 dakika cache'ler.
- **İlk tahmin yavaş**: CPU'da 330M model, 512 uzunluklu context ile birkaç
  saniye sürer. Backtest tek `predict_batch` çağrısında toplu çalışır.

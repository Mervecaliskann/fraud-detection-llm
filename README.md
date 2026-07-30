# Fraud Detection & AML Copilot

Banka işlemlerinde dolandırıcılığı (fraud) tespit eden ve her şüpheli işlem için sade Türkçe bir "Şüpheli İşlem Raporu" taslağı üreten uçtan uca bir sistem. Böylece bir uyum (AML) müfettişi hem tespiti hem de gerekçesini inceleyip onaylayabileceği bir formatta alır.

*An end-to-end system that flags fraudulent bank transactions and drafts a plain-Turkish "Suspicious Transaction Report" for each — giving a compliance investigator both the detection and the reasoning in a reviewable form.*

## Neden bu repo var / Why this repo exists

Sadece "bu işlem şüpheli" diyen bir model işin yarısını yapar. Bir bankada müfettişin hâlâ *neden* şüpheli olduğunu anlaması ve resmi bir rapor yazması gerekir — yavaş, manuel bir iş. Bu proje tespit modelini bir açıklama katmanıyla birleştirir: **XGBoost** şüpheliyi bulur, **SHAP** nedenini açıklar, **LLM** bunu Türkçe rapor taslağına çevirir.

Veri, herkese açık [PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) sentetik mobil ödeme veri setidir. Gerçek müşteri verisi kullanılmamıştır.

*A model that just says "this is fraud" does half the job — an investigator still has to understand why and write a report. This pairs detection with explanation: XGBoost finds it, SHAP explains it, an LLM drafts the Turkish report. Data is the public PaySim synthetic dataset; no real customer data.*

## Ne yapıyor / What it does

### 1. Veri keşfi / Data exploration
Fraud çok nadir (~%0.13), yani bu bir dengesiz sınıflandırma problemi — bu hem model seçimini hem metrikleri belirler. Önemli bir bulgu: fraud **yalnızca TRANSFER ve CASH_OUT** işlemlerinde görülüyor, ödeme veya para girişinde değil. Mantıklı (dolandırıcı parayı *çıkarır*) ve modeli bu iki tipe odaklamamızı sağlıyor.

*Fraud is rare (~0.13%) → imbalanced problem. Key finding: fraud occurs only in TRANSFER and CASH_OUT, which focuses the model.*

### 2. Özellik türetme / Feature engineering
Ham sütunlar tek başına yetmez. Normal bir işlemde `eski bakiye − yeni bakiye = tutar` olmalı; tutmuyorsa bir tuhaflık var. İki türetilmiş özellik bunu yakalar: `errorBalanceOrig` (gönderen tutarsızlığı) ve `errorBalanceDest` (alıcı tutarsızlığı). Fraud işlemlerde hesap sıfırlanır, bu özellikler tam onu yakalar. ID sütunları atılır (leakage riski), işlem tipi ikili kodlanır.

*Derived balance-inconsistency features catch the account being emptied; ID columns dropped (leakage); type binary-encoded.*

### 3. Tespit / Detection (XGBoost)
XGBoost tablo verisi için güçlü — doğrusal bir modelin kaçıracağı ilişkileri yakalar. Dengesizlik `scale_pos_weight` ile, bölme `stratify` ile ele alınır. Accuracy bilerek kullanılmaz ("hepsi normal" diyen model %99.9 alır, tek fraud yakalamaz).

| Metrik / Metric | Skor / Score |
|-----------------|--------------|
| PR-AUC | 0.997 |
| ROC-AUC | 0.998 |
| Recall (fraud) | 0.997 |
| Precision (fraud) | 0.981 |

Recall 0.997 → gerçek fraud'ların ~%99.7'si yakalanıyor; precision 0.981 → yanlış alarm düşük.  

### 4. Açıklanabilirlik / Explainability (SHAP)
SHAP her tahmini özellik katkılarına ayırır. İki amaç: modelin tek bir sızıntılı özelliğe değil birden çok sinyale dayandığını doğrulamak, ve tek bir işlem için "neden" faktörlerini çıkarmak. En güçlü sinyal, türetilen `errorBalanceOrig` (bakiye tutarsızlığı), ardından işlem tutarı ve hesabın sıfırlanması.

*SHAP validates the model uses several signals (not one leaky feature) and extracts the "why". Strongest signal: the derived errorBalanceOrig.*

### 5. SHAP → LLM: bayraktan rapora / from a flag to a report
En yüksek riskli işlem için SHAP faktörleri ve işlem detayları bir LLM'e (Groq üzerinden Llama-3.3) verilir; model teknik terim içermeyen, resmi Türkçe bir Şüpheli İşlem Raporu taslağı üretir.

**Örnek Çıktı / Example Output:**

> 🚨 **ŞÜPHELİ İŞLEM BİLDİRİMİ (ŞİB) - RİSK SEVİYESİ: Kritik**
>
> **📌 İşlem Özeti**
> * **İşlem Tipi:** TRANSFER
> * **İşlem Tutarı:** 7,703,574.71 TL
> * **AI Fraud Skoru:** %100.0
>
> **⚠️ Tespit Edilen Risk Faktörleri**
> * **Gönderen bakiye tutarsızlığı:** Gönderen hesabının işlem öncesi ve sonrası bakiyeleri arasında tutarsızlık olması, hesabın sıfırlanması ve yüksek tutarlı işlem yapılması
> * **Yüksek işlem tutarı:** 7,703,574.71 TL gibi yüksek bir tutarın transfer edilmesi
> * **Gönderen hesabının sıfırlanması:** İşlem sonrasında gönderen hesabının bakiyesinin sıfırlanması
>
> 📝 **AI Aksiyon Önerisi:** İşlem derhal bloke edilmeli ve detaylı bir şekilde incelenmelidir.

Bu zincir — **model (bayrak) → SHAP (neden) → LLM (okunabilir rapor)** — projenin amacıdır.

## Kurulum / Setup

Google Colab için hazırlandı (GPU gerekmez; LLM adımı hosted API kullanır).

1. PaySim CSV'yi [Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)'dan indirip notebook'a yükle.
2. Colab **Secrets**'a `GROQ_API_KEY` ekle.
3. `Runtime → Run all`.

```bash
pip install pandas numpy scikit-learn xgboost shap groq
```

## Teknik yığın / Tech stack
scikit-learn · XGBoost · SHAP (TreeExplainer) · Groq / Llama-3.3 · pandas / NumPy

## Bilinen sınırlar / Known limitations
- **Sentetik veri / Synthetic data:** PaySim gerçek değil; gerçek fraud desenleri daha karmaşıktır.
- **Türetilen bakiye özellikleri güçlü sinyal veriyor** — yüksek skoru kısmen açıklar; gerçekte tek bir özellik bu kadar belirleyici olmaz.
- **LLM raporu taslaktır, resmi belge değildir** — müfettiş incelemesi gerekir.
- **Tek train/test split**, hiperparametre optimizasyonu yok — çalışan prototip, tune edilmiş production modeli değil.

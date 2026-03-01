# e2-siap-2025

Projekat za predikciju dnevnog trenda cene zlata pomoću Random Forest klasifikatora.

## 1 minut pitch (za početak odbrane)

U ovom projektu razvili smo machine learning model koji predviđa dnevni trend cene zlata kroz tri klase: pad, stabilno i rast. Koristimo istorijske podatke sa Yahoo Finance iz perioda 2010-2025, uz kombinaciju tehničkih indikatora i makroekonomskih faktora, kako bismo modelu dali širi tržišni kontekst. Podatke pripremamo kroz 30-dnevne sekvence, balansiramo trening skup oversampling metodom i treniramo Random Forest sa 500 stabala.

Rezultati pokazuju da model uspešno predviđa sve tri klase i ostvaruje performanse iznad slučajnog pogađanja, što je značajno za ovako težak finansijski problem. Glavna vrednost projekta je robustan i jasan pipeline: od preuzimanja podataka, preko feature engineering-a, do evaluacije i interpretacije rezultata kroz grafikone. Kao sledeći korak planiramo poređenje sa naprednijim modelima i dodatnu validaciju na novijim out-of-sample podacima.

## Cilj projekta
s
Model predviđa 3 klase dnevnog kretanja cene zlata:
- **Pad**: promena < -0.5%
- **Stabilno**: promena između -0.5% i +0.5%
- **Rast**: promena > +0.5%

## Podaci i pristup

- Izvor podataka: Yahoo Finance
- Period: 2010-2025
- Ulazne promenljive: cena zlata + tehnički indikatori + makro faktori
- Lookback: 30 dana (sekvence)
- Podela skupa: 70% trening, 15% validacija, 15% test (hronološki)
- Balansiranje klasa: random oversampling na trening skupu

## Struktura projekta

- `Project/tradingbot.py` – glavni skript za trening i evaluaciju
- `Project/modules/data_loader.py` – preuzimanje i spajanje podataka
- `Project/modules/feature_engineering.py` – tehnički indikatori
- `Project/modules/preprocessing.py` – klase, sekvence i oversampling
- `Project/modules/models.py` – Random Forest konfiguracija
- `Project/modules/evaluation.py` – metrike i evaluacija
- `Project/Trading_Bot_Analysis.ipynb` – notebook sa grafikonima i objašnjenjima

## Pokretanje

Iz foldera `Project`:

```powershell
python tradingbot.py
```

Notebook analiza:
1. Otvoriti `Project/Trading_Bot_Analysis.ipynb`
2. Izabrati Python kernel (`.venv` ako postoji)
3. Pokrenuti ćelije redom (`Shift + Enter`)

## Kako prezentovati projekat (kratak vodič)

### 1) Problem i cilj (30-45 sekundi)
- Objasniti da je cilj klasifikacija dnevnog trenda cene zlata u 3 klase.
- Naglasiti da je finansijska predikcija teška zbog volatilnosti i šuma.

### 2) Podaci i feature engineering (45-60 sekundi)
- Prikazati da su korišćeni i tržišni i makro podaci.
- Izdvojiti tehničke indikatore (MA, RSI, MACD, ATR, Bollinger, OBV).
- Objasniti zašto je lookback od 30 dana koristan za kontekst.

### 3) Model i trening (45 sekundi)
- Ukratko predstaviti Random Forest (500 stabala, `max_depth=20`).
- Naglasiti oversampling i zašto je važan kod nebalansiranih klasa.

### 4) Rezultati i grafici (60-90 sekundi)
- **Distribucija klasa**: klasa Stabilno je najčešća, ali model predviđa sve 3 klase.
- **Confusion matrix**: najviše grešaka je između Stabilno i ekstremnih klasa (Pad/Rast).
- **Feature importance**: najviše doprinose recentni cenovni i volumenski signali.

### 5) Zaključak i dalje unapređenje (30-45 sekundi)
- Rezultat je dobar proof-of-concept za ovaj tip problema.
- Sledeći koraci: dodatni feature-i, poređenje sa XGBoost/LSTM i out-of-sample test.

## Kratka poruka za kraj prezentacije

Model ne služi kao garant profita, već kao robustan analitički okvir koji pomaže u donošenju odluka uz podatke i jasnu evaluaciju performansi.
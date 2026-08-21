#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phoenix Signal Tracker - تولید سیگنال‌های معاملاتی
اجرا در GitHub Actions
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("=" * 60)
print("🚀 Phoenix Signal Tracker - شروع اجرا")
print(f" زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ---------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------
EXTENDED_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
    'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOGEUSDT',
    'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'TRXUSDT'
]

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "signals.json")
MIN_CONFIDENCE = 45.0

# ---------------------------------------------------------
# دریافت داده‌ها از MEXC API (اصلاح‌شده)
# ---------------------------------------------------------
def get_mexc_candles(symbol, interval='1d', limit=200):
    """دریافت کندل‌ها از API MEXC - نسخه اصلاح‌شده"""
    url = "https://api.mexc.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️ خطای HTTP {response.status_code} برای {symbol}")
            return None
        
        data = response.json()
        
        # چک کردن اینکه data آرایه است و خالی نیست
        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠️ داده‌ای برای {symbol} یافت نشد")
            return None
        
        # بررسی ساختار داده - MEXC ممکنه ۸ یا ۱ ستون برگردونه
        first_candle = data[0]
        print(f"📊 {symbol}: تعداد ستون‌ها = {len(first_candle)}")
        
        if len(first_candle) >= 8:
            # ساختار ۸ ستونی یا بیشتر
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume'
            ])
            
            # اگر ۱۲ ستون هست، بقیه رو هم اضافه کن
            if len(first_candle) >= 12:
                df.columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ]
            
            # تبدیل ستون‌های عددی
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        return None
        
    except Exception as e:
        print(f"️ خطا در دریافت {symbol}: {e}")
        return None

print("\n🔄 در حال دریافت داده‌های تاریخی از MEXC...")
all_data = []
for sym in EXTENDED_SYMBOLS:
    df = get_mexc_candles(sym, interval='1d', limit=200)
    if df is not None:
        all_data.append(df)
        print(f"✅ {sym}: {len(df)} کندل دریافت شد")
    else:
        print(f"❌ {sym}: دریافت داده با شکست مواجه شد")
    time.sleep(0.3)

# چک کردن اینکه داده‌ای داریم یا نه
if len(all_data) == 0:
    print("\n❌ هیچ داده‌ای دریافت نشد! بررسی کنید:")
    print("  1. آیا API MEXC در دسترس است؟")
    print("  2. آیا فرمت نمادها درست است؟ (BTCUSDT نه BTC-USDT)")
    print("  3. آیا به اینترنت دسترسی دارید؟")
    exit(1)

final_df = pd.concat(all_data, ignore_index=True)
print(f"\n📊 دیتاست نهایی: {len(final_df)} ردیف")

# ---------------------------------------------------------
# Feature Engineering پیشرفته
# ---------------------------------------------------------
def calculate_advanced_features(df):
    """محاسبه فیچرهای پیشرفته"""
    
    # میانگین‌های متحرک
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volume Profile
    df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / (df['Volume_SMA'] + 1e-10)
    
    # تغییرات قیمت
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['price_change_3d'] = df['close'].pct_change(3) * 100
    
    # فاصله از SMA
    df['dist_from_SMA30'] = (df['close'] - df['SMA_30']) / df['SMA_30'] * 100
    
    return df.dropna().reset_index(drop=True)

print("\n⚙️ در حال محاسبه فیچرهای تکنیکال پیشرفته...")
processed_chunks = []
for sym, group in final_df.groupby('symbol'):
    processed_group = calculate_advanced_features(group.copy())
    processed_chunks.append(processed_group)

features_df = pd.concat(processed_chunks, ignore_index=True)
print(f"✅ فیچرها محاسبه شدند. ابعاد: {features_df.shape}")

# ---------------------------------------------------------
# محاسبه Relative Strength
# ---------------------------------------------------------
def calculate_relative_strength(df):
    """محاسبه قدرت نسبی نسبت به BTC"""
    btc_data = df[df['symbol'] == 'BTCUSDT'][['timestamp', 'price_change_pct']].rename(
        columns={'price_change_pct': 'btc_change'}
    )
    
    df = df.merge(btc_data, on='timestamp', how='left')
    df['relative_strength'] = df['price_change_pct'] - df['btc_change']
    
    return df

features_df = calculate_relative_strength(features_df)
print("✅ Relative Strength محاسبه شد")

# ---------------------------------------------------------
# برچسب‌گذاری و آموزش مدل
# ---------------------------------------------------------
def label_data_advanced(df):
    """برچسب‌گذاری پیشرفته"""
    df['future_close_3d'] = df['close'].shift(-3)
    
    buy_condition = df['future_close_3d'] > (df['close'] + 1.5 * df['ATR'])
    sell_condition = df['future_close_3d'] < (df['close'] - 1.5 * df['ATR'])
    
    df['target'] = 0
    df.loc[buy_condition, 'target'] = 1
    df.loc[sell_condition, 'target'] = 2
    
    return df.dropna().reset_index(drop=True)

print("\n🎯 در حال برچسب‌گذاری و آموزش مدل...")
labeled_chunks = []
for sym, group in features_df.groupby('symbol'):
    labeled_group = label_data_advanced(group.copy())
    labeled_chunks.append(labeled_group)

dataset_ready = pd.concat(labeled_chunks, ignore_index=True)

feature_cols = [
    'SMA_10', 'SMA_30', 'SMA_50', 'RSI', 'ATR',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volume_Ratio', 'price_change_pct', 'price_change_3d',
    'dist_from_SMA30', 'relative_strength'
]

X = dataset_ready[feature_cols]
y = dataset_ready['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n🎯 دقت مدل: {accuracy_score(y_test, y_pred):.2%}")

# ---------------------------------------------------------
# تولید سیگنال‌های زنده
# ---------------------------------------------------------
print("\n🚀 در حال تولید سیگنال‌های زنده...")
live_signals = []

for sym in EXTENDED_SYMBOLS:
    coin_data = features_df[features_df['symbol'] == sym].copy()
    
    if len(coin_data) == 0:
        continue
    
    last_row = coin_data.iloc[-1]
    
    features_input = pd.DataFrame(
        [[last_row[col] for col in feature_cols]], 
        columns=feature_cols
    )
    
    prob = model.predict_proba(features_input)[0]
    
    current_price = last_row['close']
    atr_val = last_row['ATR']
    
    buy_prob = prob[1] * 100
    sell_prob = prob[2] * 100
    
    if buy_prob >= sell_prob and buy_prob >= MIN_CONFIDENCE:
        action = "BUY (Long)"
        max_chance = buy_prob
        entry_price = current_price * 0.995
        stop_loss = entry_price - (1.5 * atr_val)
        take_profit = entry_price + (2.5 * atr_val)
    elif sell_prob > buy_prob and sell_prob >= MIN_CONFIDENCE:
        action = "SELL (Short)"
        max_chance = sell_prob
        entry_price = current_price * 1.005
        stop_loss = entry_price + (1.5 * atr_val)
        take_profit = entry_price - (2.5 * atr_val)
    else:
        continue
    
    live_signals.append({
        'symbol': sym.replace('USDT', '-USDT'),
        'current_price': round(current_price, 4),
        'action': action,
        'signal_confidence %': round(max_chance, 2),
        'entry': round(entry_price, 4),
        'stop_loss': round(stop_loss, 4),
        'take_profit': round(take_profit, 4),
        'rsi': round(last_row['RSI'], 2),
        'macd_hist': round(last_row['MACD_Hist'], 4),
        'volume_ratio': round(last_row['Volume_Ratio'], 2),
        'relative_strength': round(last_row['relative_strength'], 2)
    })

# چک کردن اینکه سیگنالی تولید شده یا نه
if len(live_signals) == 0:
    print("\n⚠️ هیچ سیگنالی با اطمینان >= 55% تولید نشد")
    print("💡 مدل برای هیچ کوینی قطعیت کافی نداشته")
    
    # ساخت فایل JSON خالی
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        f.write("[]")
    
    print(f"\n💾 فایل خالی ذخیره شد: {OUTPUT_FILE}")
    exit(0)

# اگر سیگنال داریم، مرتب‌سازی کن
output_picks = pd.DataFrame(live_signals).sort_values(
    by='signal_confidence %', 
    ascending=False
).reset_index(drop=True)

print(f"\n✅ تعداد سیگنال‌ها: {len(output_picks)}")

# ---------------------------------------------------------
# ذخیره خروجی
# ---------------------------------------------------------
strong_signals = output_picks[output_picks['signal_confidence %'] >= MIN_CONFIDENCE].reset_index(drop=True)

if len(strong_signals) == 0:
    print("️ هیچ سیگنال قوی یافت نشد")
    strong_signals = pd.DataFrame()

signals_json = strong_signals.to_json(orient='records', indent=4, force_ascii=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
    f.write(signals_json)

print(f"\n💾 فایل ذخیره شد: {OUTPUT_FILE}")
print(f"📊 تعداد سیگنال‌های قوی: {len(strong_signals)}")
print("\n🎉 عملیات با موفقیت کامل شد!")


import requests
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from openai import OpenAI
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns

client= OpenAI(api_key="XXX")
global y_test,y_pred
# 🔽 Binance API-dən məlumatları çək
def fetch_binance_ohlc():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "LTCUSDT",
        "interval": "1d",
        "limit": 2000 #days 1500
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    ohlc = []
    for item in data:
        date = datetime.fromtimestamp(item[0] / 1000).date()
        open_price = float(item[1])
        high_price = float(item[2])
        low_price = float(item[3])
        close_price = float(item[4])
        volume = float(item[5])
        ohlc.append([date, open_price, high_price, low_price, close_price, volume])

    df = pd.DataFrame(ohlc, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return df

def cross_validate_model(model, X_scaled, y):
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print("\n🧪 Cross-validation nəticələri (R²):", scores)
    print(f"📊 Ortalama R²: {scores.mean():.4f}")

# 🔽 Modeli öyrət və proqnoz üçün hazırla
def train_neural_network():
    print("📊 Binance-dən məlumat yüklənir...")
    df = fetch_binance_ohlc()

    # Feature və target ayır
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(hidden_layer_sizes=(128, 128),
                         activation='relu',
                         solver='adam',
                         max_iter=10000, 
                         random_state=42)

    print("🚀 Model öyrədilir...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    cross_validate_model(model, X_scaled, y)

    print(f"🎯 R² Score: {r2:.4f}")

    return model, scaler, df,y_test,y_pred

# 🔽 İstifadəçidən tarix al və qiyməti proqnozlaşdır
def predict_price(model, scaler, df):
    year = int(input("İli daxil et (məs: 2025): "))
    month = int(input("Ayi daxil et (1-12): "))
    day = int(input("Günü daxil et (1-31): "))

    try:
        selected_date = datetime(year, month, day).date()
        row = df[df["Date"] == selected_date]

        if row.empty:
            print("❌ Bu tarix üçün məlumat tapılmadı!")
            return


        input_features = row[["Open", "High", "Low", "Volume"]].values
        input_scaled = scaler.transform(input_features)
        predicted_price = model.predict(input_scaled)[0]

        open_price = row["Open"].values[0]
        high_price = row["High"].values[0]
        low_price = row["Low"].values[0]
        volume = row["Volume"].values[0]

        print(f"\n📅 Tarix: {selected_date}")
        print(f"🤖 Predicted Price: ${predicted_price:.4f}")
        print(f"✅ Actual Price: ${row['Close'].values[0]:.4f}")
        print("\n🧠 ChatGPT Explanation:")
        explain_prediction(open_price, high_price, low_price, volume, predicted_price)


    except Exception as e:
        print(f"Xəta baş verdi: {e}")



def explain_prediction(open_price, high_price, low_price, volume, predicted_price):
    prompt = (
        f"Aşağıdakı verilmiş maliyyə dəyərlərinə əsasən Litecoin qiyməti təxmin edilib:\n"
        f"Open: ${open_price:.2f}\n"
        f"High: ${high_price:.2f}\n"
        f"Low: ${low_price:.2f}\n"
        f"Volume: {volume:.2f}\n"
        f"Təxmin edilən qiymət: ${predicted_price:.2f}\n\n"
        f"Zəhmət olmasa bu nəticəni insan üçün başa düşülən şəkildə qısa izah et."
    )

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sən maliyyə sahəsində ekspert AI köməkçisisən."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
    )

    explanation = response.choices[0].message.content
    print(explanation)

def analyze_outliers_volume():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Volume"])
    plt.title("Volume üçün Boxplot (Outlier-lər daxil)")
    plt.xlabel("Volume")
    plt.grid(True)
    plt.show()

def analyze_outliers_open():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Open"])
    plt.title("Open üçün Boxplot (Outlier-lər daxil)")
    plt.xlabel("Open")
    plt.grid(True)
    plt.show()

def analyze_outliers_high():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["High"])
    plt.title("High üçün Boxplot (Outlier-lər daxil)")
    plt.xlabel("High")
    plt.grid(True)
    plt.show()

def analyze_outliers_low():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Low"])
    plt.title("Low üçün Boxplot (Outlier-lər daxil)")
    plt.xlabel("Low")
    plt.grid(True)
    plt.show()


def residual_plot(y_test,y_pred):
    # Residual-ların hesablanması
    residuals = y_test - y_pred

    # Residual Plot (X əvəzinə proqnoz da istifadə oluna bilər)
    sns.scatterplot(x=y_pred, y=residuals,alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Proqnozlaşdırılmış qiymət (ŷ)")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residual Plot")
    plt.show()


def histogram_high():
    sns.histplot(df['High'],kde=True)
    plt.title("High dəyişəninin paylanması")
    plt.xlabel("High")
    plt.ylabel("Say")
    plt.grid(True)
    plt.show()



def histogram_open():
    sns.histplot(df['Open'],kde=True)
    plt.title("Open dəyişəninin paylanması")
    plt.xlabel("Open")
    plt.ylabel("Say")
    plt.grid(True)
    plt.show()


def histogram_low():
    sns.histplot(df['Low'],kde=True)
    plt.title("Low dəyişəninin paylanması")
    plt.xlabel("Low")
    plt.ylabel("Say")
    plt.grid(True)
    plt.show()









def correlation():
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[["Open", "High", "Low", "Close", "Volume"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Korrelyasiya Xəritəsi")
    plt.show()


# 🔽 Başlat
if __name__ == "__main__":
    model, scaler, df,y_test,y_pred = train_neural_network()
    predict_price(model, scaler, df)
    residual_plot(y_test,y_pred)
    analyze_outliers_low()
    analyze_outliers_open()
    analyze_outliers_volume()
    analyze_outliers_high()
    histogram_high()
    histogram_open()
    histogram_low()
    correlation()

# 🚀 Crypto Price Prediction using NLP & Deep Learning

A **production-ready** cryptocurrency forecasting platform that combines:
- 📊 **Historical market data** (Binance, CoinGecko)
- 🧠 **NLP Sentiment Analysis** (FinBERT transformer model)
- 🤖 **Deep Learning** (LSTM, GRU, Transformer, Hybrid)
- ⚡ **Real-time API** (FastAPI + React dashboard)

---

## 📂 Project Structure

```
├── ml/                          # Machine Learning pipeline
│   ├── data/
│   │   ├── collectors/          # Binance, CoinGecko, NewsAPI, Reddit, Twitter
│   │   └── preprocessors/       # Market cleaner, NLP text preprocessor
│   ├── features/
│   │   ├── technical_indicators.py  # RSI, MACD, BB, ATR, Stoch, OBV, CCI...
│   │   └── feature_engineering.py  # Merge market + sentiment features
│   ├── nlp/
│   │   └── sentiment_analyzer.py   # FinBERT batch inference + time aggregation
│   ├── models/
│   │   ├── lstm_model.py        # LSTM + self-attention
│   │   ├── gru_model.py         # Bidirectional GRU + layer norm
│   │   ├── transformer_model.py # Positional encoding + encoder stack
│   │   └── hybrid_model.py      # LSTM + Transformer + Sentiment MLP (Gated Fusion)
│   ├── training/
│   │   ├── train_pipeline.py    # End-to-end training script
│   │   ├── trainer.py           # Universal trainer (early stopping, checkpointing)
│   │   ├── evaluator.py         # RMSE, MAE, MAPE, R², F1, Directional Accuracy
│   │   └── optimizer.py         # Optuna hyperparameter search
│   └── backtesting/
│       └── backtester.py        # Trading strategy simulation (Sharpe, MDD, win rate)
│
├── backend/                     # FastAPI microservice
│   ├── app/
│   │   ├── main.py              # App factory, CORS, lifespan
│   │   ├── config.py            # Pydantic-settings (.env loader)
│   │   ├── database.py          # PostgreSQL (asyncpg) + MongoDB (motor)
│   │   ├── celery_app.py        # Celery tasks + beat schedule
│   │   ├── models/crypto.py     # SQLAlchemy ORM models
│   │   └── routers/             # market, sentiment, prediction, alerts
│   ├── migrations/init_db.py    # One-shot table creation
│   └── Dockerfile
│
├── frontend/                    # React dashboard
│   ├── src/
│   │   ├── App.jsx              # Router + nav
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Live prices + OHLCV chart
│   │   │   ├── PredictionPage.jsx  # Run inference + history chart
│   │   │   ├── SentimentPage.jsx   # FinBERT sentiment timeline
│   │   │   └── AlertsPage.jsx      # Create/manage price alerts
│   │   └── services/api.js      # Axios API client
│   ├── Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml           # Full stack: Postgres + Mongo + Redis + API + Worker + UI
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Collection | `python-binance`, `pycoingecko`, `newsapi-python`, `praw`, `tweepy` |
| NLP | `HuggingFace Transformers` (FinBERT/RoBERTa), `NLTK`, `spaCy`, `emoji` |
| Deep Learning | `PyTorch` (LSTM, GRU, Transformer, Hybrid) |
| Feature Engineering | `pandas-ta`, `ta`, technical indicators |
| Hyperparameter Tuning | `Optuna` |
| Backtesting | Custom event-driven engine (Sharpe, MDD, win rate) |
| Explainability | `SHAP` |
| Backend API | `FastAPI`, `SQLAlchemy` (async), `Motor`, `Celery`, `Redis` |
| Databases | `PostgreSQL` (market/predictions), `MongoDB` (sentiment) |
| Frontend | `React 18`, `Chart.js`, `Recharts`, `Tailwind CSS` |
| Deployment | `Docker Compose`, `Nginx` |

---

## ⚡ Quick Start

### 1. Clone & configure
```bash
git clone <repo-url>
cd Crypto-Price-Prediction-using-NLP-Deep-Learning
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Launch with Docker Compose
```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| React Dashboard | http://localhost:3000 |
| FastAPI Docs | http://localhost:8000/docs |
| Flower (Celery) | http://localhost:5555 |

---

## 🧪 Train a Model Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the Hybrid model on BTC daily data
python -m ml.training.train_pipeline \
  --symbol BTC \
  --interval 1d \
  --model hybrid \
  --task regression \
  --seq_len 60 \
  --epochs 150

# Available models: lstm | gru | transformer | hybrid
# Available tasks:  regression | classification
```

Model checkpoints are saved to `ml/saved_models/BTC_hybrid_regression_best.pt`

---

## 🔌 API Endpoints

### Market Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/market/price/{symbol}` | Live price from Binance |
| `GET` | `/api/v1/market/ohlcv/{symbol}` | Stored OHLCV candles |
| `GET` | `/api/v1/market/supported-symbols` | Tracked symbols list |

### Sentiment
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/sentiment/{symbol}?hours=48` | FinBERT sentiment timeline |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/prediction/predict` | Run live model inference |
| `GET` | `/api/v1/prediction/history/{symbol}` | Stored predictions vs actual |

#### Prediction request body
```json
{
  "symbol": "BTC",
  "model_name": "hybrid",
  "task": "regression",
  "horizon": 1
}
```

### Alerts
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/alerts/` | Create price/sentiment alert |
| `GET` | `/api/v1/alerts/` | List all alerts |
| `DELETE` | `/api/v1/alerts/{id}` | Remove an alert |

---

## 🧠 Models

### LSTM + Attention
Stacked LSTM with configurable layers and scaled dot-product attention over time steps. Supports regression and classification heads.

### GRU (Bidirectional)
Bidirectional GRU with layer normalisation. Lighter than LSTM, faster convergence.

### Transformer Encoder
Positional encoding + N-layer encoder with GELU activation and multi-head self-attention. Suitable for longer sequences (>100 steps).

### Hybrid (LSTM + Transformer + Sentiment Fusion) ⭐
LSTM encoder → Transformer attention layer with a separate MLP branch for aggregated sentiment features.
A **gated fusion** layer combines market context + sentiment embedding for best multi-modal performance.

---

## 📈 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression | RMSE, MAE, MAPE, R², Directional Accuracy |
| Classification | Accuracy, F1-score, Confusion Matrix |
| Trading | Sharpe Ratio, Max Drawdown, Win Rate, Annualised Return |

---

## 🔧 Hyperparameter Optimisation

```python
from ml.training.optimizer import optimise

study = optimise(
    model_type="lstm",
    input_size=45,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=50,
)
print(study.best_params)
```

---

## 🗄️ Environment Variables

Copy `.env.example` to `.env` and set:

| Variable | Description |
|----------|-------------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Binance market data |
| `COINGECKO_API_KEY` | CoinGecko (optional for free tier) |
| `NEWS_API_KEY` | NewsAPI.org articles |
| `CRYPTOPANIC_API_KEY` | CryptoPanic posts |
| `TWITTER_BEARER_TOKEN` | Twitter v2 API |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit PRAW |
| `POSTGRES_*` | PostgreSQL connection |
| `MONGO_URI` | MongoDB connection |
| `REDIS_URL` | Redis (broker + cache) |
| `HUGGINGFACE_MODEL` | Sentiment model (default: `ProsusAI/finbert`) |

---

## 🚢 Deployment

```bash
# Full stack launch
docker compose up --build -d

# View logs
docker compose logs -f backend

# Scale workers
docker compose up --scale worker=4 -d
```

For production:
- Set `APP_ENV=production` and a strong `SECRET_KEY`
- Restrict `ALLOWED_ORIGINS` to your domain
- Add HTTPS via Nginx + Let's Encrypt
- Use managed databases (AWS RDS, MongoDB Atlas)

---

## 🔒 Security

- No API keys in source code — environment variables only
- Input validation via Pydantic v2
- SQL injection prevention via SQLAlchemy ORM
- CORS restricted to configured origins

---

## 📊 Dashboard Pages

| Page | Features |
|------|----------|
| **Dashboard** | Live price cards, OHLCV + volume chart, symbol/interval switcher |
| **Predictions** | Model selector, inference runner, BUY/SELL/HOLD signal badge, history overlay |
| **Sentiment** | FinBERT score gauge, hourly sentiment line chart, positive/negative bar chart |
| **Alerts** | Create price/sentiment threshold alerts, email notifications, active alert list |

---

## 🧩 Reproducibility

All data splits use **chronological ordering** (no shuffling) to prevent look-ahead bias.
Fix seeds before training:

```python
import torch, numpy, random
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
```

---

## 📄 License

MIT License © 2024
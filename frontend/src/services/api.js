/**
 * Centralised API client using axios.
 * All backend calls go through this module.
 */
import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || '/api/v1';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Market ────────────────────────────────────────────────────────────────

export const fetchCurrentPrice = (symbol) =>
  api.get(`/market/price/${symbol}`).then((r) => r.data);

export const fetchOHLCV = (symbol, interval = '1d', limit = 200) =>
  api.get(`/market/ohlcv/${symbol}`, { params: { interval, limit } }).then((r) => r.data);

export const fetchSupportedSymbols = () =>
  api.get('/market/supported-symbols').then((r) => r.data.symbols);

// ── Sentiment ─────────────────────────────────────────────────────────────

export const fetchSentiment = (symbol, hours = 48) =>
  api.get(`/sentiment/${symbol}`, { params: { hours } }).then((r) => r.data);

// ── Predictions ───────────────────────────────────────────────────────────

export const runPrediction = (payload) =>
  api.post('/prediction/predict', payload).then((r) => r.data);

export const fetchPredictionHistory = (symbol, modelName = 'hybrid', limit = 50) =>
  api
    .get(`/prediction/history/${symbol}`, { params: { model_name: modelName, limit } })
    .then((r) => r.data);

// ── Alerts ────────────────────────────────────────────────────────────────

export const fetchAlerts = (symbol) =>
  api.get('/alerts/', { params: symbol ? { symbol } : {} }).then((r) => r.data);

export const createAlert = (payload) =>
  api.post('/alerts/', payload).then((r) => r.data);

export const deleteAlert = (id) =>
  api.delete(`/alerts/${id}`).then((r) => r.data);

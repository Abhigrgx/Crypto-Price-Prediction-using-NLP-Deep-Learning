import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import { Line } from 'react-chartjs-2';
import { runPrediction, fetchPredictionHistory } from '../services/api';

const MODELS = ['hybrid', 'lstm', 'gru', 'transformer'];
const SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'];

function SignalBadge({ signal }) {
  const colors = { BUY: 'bg-green-600', SELL: 'bg-red-600', HOLD: 'bg-yellow-600' };
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-bold text-white ${colors[signal] || 'bg-gray-600'}`}>
      {signal}
    </span>
  );
}

export default function PredictionPage() {
  const [symbol, setSymbol] = useState('BTC');
  const [model, setModel] = useState('hybrid');
  const [result, setResult] = useState(null);

  const { data: history = [] } = useQuery(
    ['pred-history', symbol, model],
    () => fetchPredictionHistory(symbol, model),
    { enabled: true }
  );

  const predictMutation = useMutation(runPrediction, {
    onSuccess: (data) => setResult(data),
  });

  const handlePredict = () => {
    predictMutation.mutate({ symbol, model_name: model, task: 'regression', horizon: 1 });
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Price Predictions</h1>

      {/* Controls */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
            >
              {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
            >
              {MODELS.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
        </div>
        <button
          onClick={handlePredict}
          disabled={predictMutation.isLoading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          {predictMutation.isLoading ? 'Running inference…' : 'Run Prediction'}
        </button>
      </div>

      {/* Result card */}
      {result && (
        <div className="bg-gray-800 rounded-xl p-6 border border-indigo-700 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">{result.symbol} – {result.model_name}</h2>
            <SignalBadge signal={result.signal} />
          </div>
          {result.predicted_price && (
            <p className="text-3xl font-bold text-indigo-300">
              ${Number(result.predicted_price).toLocaleString('en-US', { maximumFractionDigits: 2 })}
            </p>
          )}
          {result.confidence && (
            <p className="text-sm text-gray-400">
              Confidence: <span className="text-white">{(result.confidence * 100).toFixed(1)}%</span>
            </p>
          )}
          <p className="text-xs text-gray-500">
            Predicted at: {new Date(result.predicted_at).toLocaleString()}
          </p>
        </div>
      )}

      {predictMutation.isError && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300">
          {predictMutation.error?.response?.data?.detail || 'Prediction failed. Ensure the model is trained first.'}
        </div>
      )}

      {/* History chart */}
      {history.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-lg font-semibold mb-4">Prediction History vs Actual</h2>
          <Line
            data={{
              labels: history.map((h) => new Date(h.predicted_at).toLocaleDateString()),
              datasets: [
                {
                  label: 'Predicted Price',
                  data: history.map((h) => h.predicted_price),
                  borderColor: '#818cf8',
                  pointRadius: 3,
                  tension: 0.3,
                },
                {
                  label: 'Actual Price',
                  data: history.map((h) => h.actual_price),
                  borderColor: '#34d399',
                  pointRadius: 3,
                  tension: 0.3,
                },
              ],
            }}
            options={{
              responsive: true,
              scales: { x: { ticks: { color: '#9ca3af' } }, y: { ticks: { color: '#9ca3af' } } },
            }}
          />
        </div>
      )}
    </div>
  );
}

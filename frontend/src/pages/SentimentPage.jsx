import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Line, Bar } from 'react-chartjs-2';
import { fetchSentiment } from '../services/api';

const SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'];

function SentimentMeter({ score }) {
  const pct = ((score + 1) / 2) * 100;
  const color = score > 0.1 ? '#34d399' : score < -0.1 ? '#f87171' : '#fbbf24';
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>Bearish</span><span>Neutral</span><span>Bullish</span>
      </div>
      <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <p className="text-center text-sm font-semibold" style={{ color }}>
        Score: {score.toFixed(3)}
      </p>
    </div>
  );
}

export default function SentimentPage() {
  const [symbol, setSymbol] = useState('BTC');
  const [hours, setHours] = useState(48);

  const { data, isLoading, isError, error } = useQuery(
    ['sentiment', symbol, hours],
    () => fetchSentiment(symbol, hours),
    { retry: 1 }
  );

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Sentiment Analysis</h1>

      {/* Controls */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 flex flex-wrap gap-4 items-center">
        <div className="flex gap-2">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                symbol === s ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <select
          value={hours}
          onChange={(e) => setHours(Number(e.target.value))}
          className="bg-gray-700 text-gray-100 text-sm px-3 py-1 rounded-lg border border-gray-600"
        >
          {[24, 48, 72, 168].map((h) => (
            <option key={h} value={h}>{h}h</option>
          ))}
        </select>
      </div>

      {isLoading && (
        <div className="bg-gray-800 rounded-xl p-8 text-center text-gray-400 animate-pulse">
          Running FinBERT sentiment analysis…
        </div>
      )}

      {isError && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300">
          {error?.response?.data?.detail || 'Failed to load sentiment data.'}
        </div>
      )}

      {data && (
        <>
          {/* Summary */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">{data.symbol} Sentiment</h2>
              <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                data.dominant_label === 'POSITIVE' ? 'bg-green-700 text-green-100' :
                data.dominant_label === 'NEGATIVE' ? 'bg-red-700 text-red-100' :
                'bg-yellow-700 text-yellow-100'
              }`}>
                {data.dominant_label}
              </span>
            </div>
            <SentimentMeter score={data.avg_sentiment} />
            <p className="text-sm text-gray-400">
              Based on <span className="text-white font-semibold">{data.total_articles}</span> articles
              over the last <span className="text-white font-semibold">{data.period_hours}h</span>
            </p>
          </div>

          {/* Time-series charts */}
          {data.data.length > 0 && (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-6">
              <h2 className="text-lg font-semibold">Sentiment Over Time</h2>
              <Line
                data={{
                  labels: data.data.map((d) => new Date(d.timestamp).toLocaleString()),
                  datasets: [{
                    label: 'Sentiment Score',
                    data: data.data.map((d) => d.sentiment_score),
                    borderColor: '#818cf8',
                    backgroundColor: 'rgba(129,140,248,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                  }],
                }}
                options={{
                  responsive: true,
                  plugins: { legend: { display: false } },
                  scales: {
                    x: { ticks: { color: '#9ca3af', maxTicksLimit: 6 } },
                    y: { ticks: { color: '#9ca3af' }, min: -1, max: 1 },
                  },
                }}
              />
              <Bar
                data={{
                  labels: data.data.map((d) => new Date(d.timestamp).toLocaleString()),
                  datasets: [
                    {
                      label: 'Positive',
                      data: data.data.map((d) => d.sentiment_positive),
                      backgroundColor: 'rgba(52,211,153,0.7)',
                    },
                    {
                      label: 'Negative',
                      data: data.data.map((d) => d.sentiment_negative),
                      backgroundColor: 'rgba(248,113,113,0.7)',
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  scales: {
                    x: { display: false },
                    y: { ticks: { color: '#9ca3af' } },
                  },
                }}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

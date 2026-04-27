import React, { useState } from 'react';
import { useQuery } from 'react-query';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { fetchCurrentPrice, fetchOHLCV, fetchSupportedSymbols } from '../services/api';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
);

function PriceCard({ symbol }) {
  const { data, isLoading, isError } = useQuery(
    ['price', symbol],
    () => fetchCurrentPrice(symbol),
    { refetchInterval: 15_000 }
  );

  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <p className="text-xs text-gray-400 uppercase tracking-widest">{symbol}/USDT</p>
      {isLoading && <p className="text-2xl font-bold text-gray-500 animate-pulse">--</p>}
      {isError && <p className="text-2xl font-bold text-red-400">Error</p>}
      {data && (
        <p className="text-2xl font-bold text-indigo-300">
          ${Number(data.price).toLocaleString('en-US', { maximumFractionDigits: 2 })}
        </p>
      )}
    </div>
  );
}

function OHLCVChart({ symbol, interval }) {
  const { data = [], isLoading } = useQuery(
    ['ohlcv', symbol, interval],
    () => fetchOHLCV(symbol, interval, 120)
  );

  if (isLoading) return <div className="h-64 flex items-center justify-center text-gray-500">Loading chart…</div>;

  const labels = data.map((c) => new Date(c.open_time).toLocaleDateString());
  const closes = data.map((c) => c.close);
  const volumes = data.map((c) => c.volume);

  return (
    <div className="space-y-4">
      <Line
        data={{
          labels,
          datasets: [
            {
              label: `${symbol} Close Price`,
              data: closes,
              borderColor: '#818cf8',
              backgroundColor: 'rgba(129,140,248,0.1)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
            },
          ],
        }}
        options={{
          responsive: true,
          plugins: { legend: { display: false } },
          scales: { x: { ticks: { maxTicksLimit: 8, color: '#9ca3af' } }, y: { ticks: { color: '#9ca3af' } } },
        }}
      />
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: 'Volume',
              data: volumes,
              backgroundColor: 'rgba(99,102,241,0.4)',
            },
          ],
        }}
        options={{
          responsive: true,
          plugins: { legend: { display: false } },
          scales: { x: { display: false }, y: { ticks: { color: '#9ca3af' } } },
        }}
      />
    </div>
  );
}

export default function Dashboard() {
  const [symbol, setSymbol] = useState('BTC');
  const [interval, setInterval] = useState('1d');
  const { data: symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'] } = useQuery(
    'symbols', fetchSupportedSymbols
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-100">Market Dashboard</h1>

      {/* Price cards row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {symbols.slice(0, 5).map((s) => <PriceCard key={s} symbol={s} />)}
      </div>

      {/* Symbol selector + chart */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          <div className="flex gap-2">
            {symbols.map((s) => (
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
            value={interval}
            onChange={(e) => setInterval(e.target.value)}
            className="bg-gray-700 text-gray-100 text-sm px-3 py-1 rounded-lg border border-gray-600"
          >
            {['1h', '4h', '1d'].map((iv) => (
              <option key={iv} value={iv}>{iv}</option>
            ))}
          </select>
        </div>
        <OHLCVChart symbol={symbol} interval={interval} />
      </div>
    </div>
  );
}

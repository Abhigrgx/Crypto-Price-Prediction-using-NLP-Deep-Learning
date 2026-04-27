import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { fetchAlerts, createAlert, deleteAlert } from '../services/api';

const ALERT_TYPES = ['price_above', 'price_below', 'sentiment_above', 'sentiment_below'];
const SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'];

export default function AlertsPage() {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    symbol: 'BTC', alert_type: 'price_above', threshold: '', notify_email: '',
  });

  const { data: alerts = [] } = useQuery('alerts', () => fetchAlerts());

  const createMutation = useMutation(createAlert, {
    onSuccess: () => qc.invalidateQueries('alerts'),
  });

  const deleteMutation = useMutation(deleteAlert, {
    onSuccess: () => qc.invalidateQueries('alerts'),
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    createMutation.mutate({
      ...form,
      threshold: parseFloat(form.threshold),
      notify_email: form.notify_email || null,
    });
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Price & Sentiment Alerts</h1>

      {/* Create form */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-lg font-semibold mb-4">Create Alert</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Symbol</label>
              <select
                value={form.symbol}
                onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
              >
                {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Alert Type</label>
              <select
                value={form.alert_type}
                onChange={(e) => setForm({ ...form, alert_type: e.target.value })}
                className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
              >
                {ALERT_TYPES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Threshold</label>
              <input
                type="number"
                step="any"
                required
                value={form.threshold}
                onChange={(e) => setForm({ ...form, threshold: e.target.value })}
                className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
                placeholder="e.g. 70000"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Email (optional)</label>
              <input
                type="email"
                value={form.notify_email}
                onChange={(e) => setForm({ ...form, notify_email: e.target.value })}
                className="w-full bg-gray-700 text-gray-100 px-3 py-2 rounded-lg border border-gray-600"
                placeholder="you@example.com"
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={createMutation.isLoading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold py-2 rounded-lg transition-colors"
          >
            {createMutation.isLoading ? 'Creating…' : 'Create Alert'}
          </button>
        </form>
      </div>

      {/* Alert list */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-lg font-semibold mb-4">Active Alerts</h2>
        {alerts.length === 0 && <p className="text-gray-500 text-sm">No alerts configured.</p>}
        <ul className="space-y-3">
          {alerts.map((alert) => (
            <li
              key={alert.id}
              className="flex items-center justify-between bg-gray-700 rounded-lg px-4 py-3"
            >
              <div>
                <span className="font-semibold text-indigo-300">{alert.symbol}</span>
                <span className="mx-2 text-gray-400 text-sm">{alert.alert_type}</span>
                <span className="text-white font-mono">{alert.threshold}</span>
                {alert.notify_email && (
                  <span className="ml-2 text-xs text-gray-400">→ {alert.notify_email}</span>
                )}
              </div>
              <button
                onClick={() => deleteMutation.mutate(alert.id)}
                className="text-red-400 hover:text-red-300 text-sm font-medium"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import Dashboard from './pages/Dashboard';
import PredictionPage from './pages/PredictionPage';
import SentimentPage from './pages/SentimentPage';
import AlertsPage from './pages/AlertsPage';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, staleTime: 30_000 } },
});

const NAV_LINKS = [
  { to: '/', label: '📊 Dashboard' },
  { to: '/prediction', label: '🔮 Predictions' },
  { to: '/sentiment', label: '🧠 Sentiment' },
  { to: '/alerts', label: '🔔 Alerts' },
];

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-950 text-gray-100">
          {/* ── Navbar ──────────────────────────────────────── */}
          <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-8">
            <span className="text-xl font-bold text-indigo-400">₿ CryptoPredict</span>
            <div className="flex gap-6">
              {NAV_LINKS.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    `text-sm font-medium transition-colors ${
                      isActive ? 'text-indigo-400' : 'text-gray-400 hover:text-gray-100'
                    }`
                  }
                >
                  {label}
                </NavLink>
              ))}
            </div>
          </nav>

          {/* ── Page content ────────────────────────────────── */}
          <main className="p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/prediction" element={<PredictionPage />} />
              <Route path="/sentiment" element={<SentimentPage />} />
              <Route path="/alerts" element={<AlertsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

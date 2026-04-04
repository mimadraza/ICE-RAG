const BASE = 'http://localhost:8000/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.detail || JSON.stringify(json));
  return json;
}

export const api = {
  health: () => request('/health'),
  info:   () => request('/info'),

  query: (payload) =>
    request('/query', { method: 'POST', body: JSON.stringify(payload) }),

  queryDefault: (q) =>
    request(`/query/default?q=${encodeURIComponent(q)}`),

  experimentsSummary: () => request('/experiments/summary'),
  chunkingReport:     () => request('/experiments/chunking-report'),

  runExperiment: (payload) =>
    request('/experiments/run', { method: 'POST', body: JSON.stringify(payload) }),
};

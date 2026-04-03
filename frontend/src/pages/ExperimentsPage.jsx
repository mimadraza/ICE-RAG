import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Card, Tag, Spinner, ErrorBox, SectionLabel } from '../components/UI';

export default function ExperimentsPage() {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    api.experimentsSummary()
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState />;
  if (error)   return <div style={{ padding: 24 }}><ErrorBox message={error} /></div>;
  if (!data?.results?.length) return <EmptyState />;

  const rows = data.results;

  // Compute best per metric
  const maxFaith = Math.max(...rows.map(r => parseFloat(r.faithfulness)));
  const maxRel   = Math.max(...rows.map(r => parseFloat(r.relevancy)));
  const bestRow  = rows.reduce((a, b) =>
    parseFloat(b.faithfulness) > parseFloat(a.faithfulness) ? b : a, rows[0]);

  const avgTime = (rows.reduce((s, r) => s + parseFloat(r.total_time), 0) / rows.length).toFixed(1);
  const bestRelRow = rows.reduce((a, b) => parseFloat(b.relevancy) > parseFloat(a.relevancy) ? b : a, rows[0]);

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: 24 }}>

      <div>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Experiment Results</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Faithfulness & relevancy scores across all pipeline configurations. Best config highlighted.
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
        {[
          { label: 'Best Faithfulness', value: `${(maxFaith * 100).toFixed(1)}%`, sub: `${bestRow.chunking} · ${bestRow.retrieval}` },
          { label: 'Best Relevancy',    value: `${(maxRel * 100).toFixed(1)}%`,   sub: `${bestRelRow.chunking} · ${bestRelRow.retrieval}` },
          { label: 'Configs Tested',    value: rows.length,                         sub: 'all strategies' },
          { label: 'Avg Total Time',    value: `${avgTime}s`,                       sub: 'per query incl. eval' },
        ].map(m => (
          <MetricCard key={m.label} {...m} />
        ))}
      </div>

      {/* Results table */}
      <Card>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 12 }}>
          <thead>
            <tr style={{ background: 'var(--surface2)' }}>
              {['Chunking', 'Retrieval', 'Rerank', 'Faithfulness', 'Relevancy', 'Total Time', ''].map(h => (
                <th key={h} style={{
                  padding: '10px 14px', textAlign: 'left',
                  color: 'var(--text-muted)', fontSize: 10,
                  letterSpacing: '0.08em', textTransform: 'uppercase',
                  fontWeight: 500, borderBottom: '1px solid var(--border)',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isBest = row === bestRow;
              const faith  = parseFloat(row.faithfulness);
              const rel    = parseFloat(row.relevancy);
              return (
                <tr key={i} style={{ background: isBest ? 'var(--accent-dim)' : 'transparent' }}>
                  <Td bold={isBest}>{row.chunking}</Td>
                  <Td bold={isBest}>{row.retrieval}</Td>
                  <Td bold={isBest}>{row.rerank}</Td>
                  <Td bold={isBest}>
                    <BarCell value={faith} max={maxFaith} color="var(--accent)" />
                  </Td>
                  <Td bold={isBest}>
                    <BarCell value={rel} max={maxRel} color="var(--accent2)" />
                  </Td>
                  <Td bold={isBest}>{parseFloat(row.total_time).toFixed(2)}s</Td>
                  <Td>
                    {isBest && <Tag variant="green">★ best</Tag>}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function Td({ children, bold }) {
  return (
    <td style={{
      padding: '10px 14px',
      borderBottom: '1px solid var(--border)',
      color: bold ? 'var(--text)' : 'var(--text-dim)',
    }}>
      {children}
    </td>
  );
}

function BarCell({ value, max, color }) {
  const pct = (value / max * 80).toFixed(0);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: `${pct}px`, height: 6, borderRadius: 3, background: color, opacity: 0.7, minWidth: 4 }} />
      {(value * 100).toFixed(1)}%
    </div>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--r)', padding: '14px 16px',
    }}>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 22, fontWeight: 500, color: 'var(--accent)' }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{sub}</div>
    </div>
  );
}

function LoadingState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 10, color: 'var(--text-muted)', fontSize: 13 }}>
      <Spinner /> Loading experiment data…
    </div>
  );
}

function EmptyState() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-muted)', fontSize: 13 }}>
      No experiment data found.
    </div>
  );
}

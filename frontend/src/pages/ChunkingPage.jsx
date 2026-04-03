import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Card, Spinner, ErrorBox } from '../components/UI';

export default function ChunkingPage() {
  const [report, setReport]   = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    api.chunkingReport()
      .then(r => setReport(r.report || ''))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Chunking Report</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Statistics for fixed, recursive, and semantic chunking strategies.
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-muted)', fontSize: 13 }}>
          <Spinner /> Loading…
        </div>
      )}

      {error && <ErrorBox message={error} />}

      {report && (
        <Card>
          <pre style={{
            padding: '20px',
            fontFamily: 'var(--mono)',
            fontSize: 12,
            color: 'var(--text-dim)',
            lineHeight: 1.7,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            margin: 0,
          }}>
            {report}
          </pre>
        </Card>
      )}
    </div>
  );
}

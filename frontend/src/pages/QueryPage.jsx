import React, { useState, useRef } from 'react';
import { api } from '../api/client';
import AnswerCard from '../components/AnswerCard';
import { Button, Spinner, ErrorBox } from '../components/UI';

export default function QueryPage({ config }) {
  const [query, setQuery]     = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const textareaRef = useRef(null);

  const handleSubmit = async () => {
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);

    try {
      const res = await api.query({
        query: q,
        chunking:   config.chunking,
        retrieval:  config.retrieval,
        rerank:     config.rerank,
        retrieve_k: config.retrieveK,
        final_k:    config.finalK,
      });
      setResults(prev => [res.data, ...prev]);
      setQuery('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextarea = (e) => {
    setQuery(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Input area */}
      <div style={{
        padding: '20px 32px',
        borderBottom: '2px solid var(--border)',
        background: 'var(--surface)',
      }}>
        <div style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          letterSpacing: '0.25em',
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          marginBottom: 10,
        }}>
          ▶ ЗАПРОС / QUERY INPUT
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end' }}>
          <div style={{
            flex: 1,
            border: '2px solid var(--border)',
            position: 'relative',
          }}>
            <div style={{
              position: 'absolute',
              top: 0, left: 0,
              width: 3,
              height: '100%',
              background: 'var(--accent)',
            }} />
            <textarea
              ref={textareaRef}
              value={query}
              onChange={handleTextarea}
              onKeyDown={handleKeyDown}
              placeholder="Введите запрос…  (Enter — отправить, Shift+Enter — новая строка)"
              rows={2}
              style={{
                width: '100%',
                background: 'var(--surface2)',
                border: 'none',
                color: 'var(--text)',
                fontFamily: 'var(--mono)',
                fontSize: 13,
                padding: '12px 14px 12px 18px',
                resize: 'none',
                outline: 'none',
                lineHeight: 1.6,
                minHeight: 54,
                display: 'block',
              }}
            />
          </div>
          <Button
            onClick={handleSubmit}
            disabled={!query.trim() || loading}
            style={{ minWidth: 100, height: 54, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          >
            {loading ? <Spinner size={13} /> : 'ИСКАТЬ'}
          </Button>
        </div>
      </div>

      {/* Results */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '24px 32px',
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        {error && <ErrorBox message={error} />}

        {results.length === 0 && !loading && (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            gap: 16,
            marginTop: 80,
            userSelect: 'none',
          }}>
            <div style={{
              fontSize: 52,
              color: 'var(--border-hi)',
              fontFamily: 'var(--sans)',
              fontWeight: 700,
              letterSpacing: '0.1em',
            }}>★</div>
            <div style={{
              fontFamily: 'var(--sans)',
              fontSize: 16,
              fontWeight: 600,
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
            }}>
              ОЖИДАНИЕ ЗАПРОСА
            </div>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'var(--border-hi)',
              letterSpacing: '0.12em',
            }}>
              {config.chunking} · {config.retrieval} · rerank {config.rerank ? 'ON' : 'OFF'}
            </div>
          </div>
        )}

        {results.map((r, i) => (
          <AnswerCard key={i} result={r} />
        ))}
      </div>
    </div>
  );
}

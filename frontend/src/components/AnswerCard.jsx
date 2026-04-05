import React, { useState } from 'react';
import { Tag } from './UI';

export default function AnswerCard({ result }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const { query, answer, config, timing, sources } = result;

  return (
    <div style={{
      background: 'var(--surface)',
      border: '2px solid var(--border)',
      borderLeft: '5px solid var(--accent)',
      animation: 'fadeUp 0.2s ease',
      overflow: 'hidden',
    }}>

      {/* Header bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 16px',
        background: 'var(--surface2)',
        borderBottom: '1px solid var(--border)',
        gap: 10,
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            letterSpacing: '0.2em',
            color: 'var(--accent)',
            marginRight: 4,
          }}>★</span>
          <Tag variant="red">{config.chunking}</Tag>
          <Tag variant="amber">{config.retrieval}</Tag>
          {config.rerank
            ? <Tag variant="red">RERANK ✓</Tag>
            : <Tag variant="default">NO RERANK</Tag>
          }
        </div>
        <div style={{
          fontFamily: 'var(--mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
          letterSpacing: '0.1em',
        }}>
          {sources?.length || 0} SOURCES
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: '16px 20px' }}>

        {/* Query echo */}
        <div style={{
          fontFamily: 'var(--mono)',
          fontSize: 12,
          color: 'var(--text-dim)',
          padding: '8px 12px 8px 16px',
          marginBottom: 14,
          background: 'var(--surface3)',
          borderLeft: '3px solid var(--accent)',
          lineHeight: 1.55,
          letterSpacing: '0.03em',
        }}>
          {query}
        </div>

        {/* Answer */}
        <div style={{
          fontSize: 15,
          lineHeight: 1.8,
          color: 'var(--text)',
          fontFamily: 'var(--sans)',
          fontWeight: 400,
          letterSpacing: '0.02em',
        }}>
          {answer}
        </div>
      </div>

      {/* Sources toggle */}
      {sources?.length > 0 && (
        <>
          <div
            onClick={() => setSourcesOpen(o => !o)}
            style={{
              padding: '9px 20px',
              borderTop: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              fontFamily: 'var(--mono)',
              fontSize: 10,
              letterSpacing: '0.15em',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              userSelect: 'none',
              textTransform: 'uppercase',
            }}
          >
            <span style={{
              display: 'inline-block',
              transition: 'transform 0.15s',
              transform: sourcesOpen ? 'rotate(90deg)' : 'none',
              color: 'var(--accent)',
            }}>▶</span>
            SOURCES ({sources.length})
          </div>

          {sourcesOpen && (
            <div style={{ padding: '4px 20px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {sources.map((s, i) => (
                <SourceItem key={i} index={i + 1} source={s} />
              ))}
            </div>
          )}
        </>
      )}

      {/* Timing footer */}
      <div style={{
        display: 'flex',
        gap: 24,
        padding: '8px 20px',
        borderTop: '1px solid var(--border)',
        background: 'var(--surface2)',
        fontFamily: 'var(--mono)',
        fontSize: 10,
        color: 'var(--text-muted)',
        letterSpacing: '0.12em',
      }}>
        {[
          ['RETRIEVAL', timing.retrieval_ms],
          ['GENERATION', timing.generation_ms],
          ['TOTAL', timing.total_ms],
        ].map(([label, ms]) => (
          <span key={label}>
            {label}{' '}
            <span style={{ color: 'var(--accent2)' }}>{ms}ms</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function SourceItem({ index, source }) {
  const score = source.rerank_score ?? source.rrf_score ?? source.score ?? 0;
  return (
    <div style={{
      background: 'var(--surface2)',
      border: '1px solid var(--border)',
      borderLeft: '3px solid var(--border-hi)',
      padding: '10px 14px',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        marginBottom: 6,
        fontFamily: 'var(--mono)',
        fontSize: 10,
        color: 'var(--text-muted)',
        letterSpacing: '0.1em',
      }}>
        <span style={{ color: 'var(--accent)', fontWeight: 500 }}>#{index}</span>
        <span style={{ color: 'var(--text-dim)' }}>{source.source || 'UNKNOWN'}</span>
        <Tag>{source.retrieval_type}</Tag>
        <span style={{ marginLeft: 'auto', color: 'var(--accent2)' }}>
          {typeof score === 'number' ? score.toFixed(3) : '—'}
        </span>
      </div>
      <div style={{
        fontSize: 11,
        color: 'var(--text-dim)',
        lineHeight: 1.6,
        fontFamily: 'var(--mono)',
      }}>
        {source.text}
      </div>
    </div>
  );
}

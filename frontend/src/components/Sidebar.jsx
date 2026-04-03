import React from 'react';
import { SectionLabel, Toggle, Select, Tag } from './UI';

const NAV_ITEMS = [
  { id: 'query',       icon: '⌖', label: 'Query' },
  { id: 'experiments', icon: '◈', label: 'Experiments' },
  { id: 'chunking',    icon: '⊞', label: 'Chunking Report' },
];

export default function Sidebar({ activePanel, onPanelChange, config, onConfigChange, bestConfig, apiStatus }) {
  return (
    <aside style={{
      background: 'var(--surface)',
      borderRight: '1px solid var(--border)',
      padding: '20px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 28,
      overflowY: 'auto',
      width: 260,
      flexShrink: 0,
    }}>

      {/* Nav */}
      <div>
        <SectionLabel>Views</SectionLabel>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {NAV_ITEMS.map(item => (
            <div
              key={item.id}
              onClick={() => onPanelChange(item.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 12px', borderRadius: 8,
                fontSize: 13, cursor: 'pointer',
                color: activePanel === item.id ? 'var(--accent)' : 'var(--text-dim)',
                background: activePanel === item.id ? 'var(--accent-dim)' : 'transparent',
                transition: 'background var(--transition), color var(--transition)',
                userSelect: 'none',
              }}
            >
              <span style={{ width: 20, textAlign: 'center', fontSize: 15 }}>{item.icon}</span>
              {item.label}
            </div>
          ))}
        </nav>
      </div>

      {/* Pipeline Config */}
      <div>
        <SectionLabel>Pipeline Config</SectionLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          <ConfigItem label="Chunking Strategy">
            <Select
              value={config.chunking}
              onChange={v => onConfigChange('chunking', v)}
              options={[
                { value: 'recursive', label: 'recursive' },
                { value: 'fixed',     label: 'fixed' },
                { value: 'semantic',  label: 'semantic' },
              ]}
            />
          </ConfigItem>

          <ConfigItem label="Retrieval Mode">
            <Select
              value={config.retrieval}
              onChange={v => onConfigChange('retrieval', v)}
              options={[
                { value: 'hybrid',   label: 'hybrid (BM25 + semantic)' },
                { value: 'semantic', label: 'semantic only' },
              ]}
            />
          </ConfigItem>

          <ConfigItem label="Reranking">
            <div style={{
              background: 'var(--surface2)', border: '1px solid var(--border)',
              borderRadius: 'var(--r)', padding: '7px 10px',
            }}>
              <Toggle
                checked={config.rerank}
                onChange={v => onConfigChange('rerank', v)}
                label="CrossEncoder rerank"
              />
            </div>
          </ConfigItem>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <ConfigItem label="Retrieve K">
              <Select
                value={config.retrieveK}
                onChange={v => onConfigChange('retrieveK', parseInt(v))}
                options={[5,10,20].map(n => ({ value: n, label: `${n} docs` }))}
              />
            </ConfigItem>
            <ConfigItem label="Final K">
              <Select
                value={config.finalK}
                onChange={v => onConfigChange('finalK', parseInt(v))}
                options={[3,5,8].map(n => ({ value: n, label: `${n} docs` }))}
              />
            </ConfigItem>
          </div>
        </div>
      </div>

      {/* Best Config Badge */}
      {bestConfig && (
        <div>
          <SectionLabel>Best Config (experiments)</SectionLabel>
          <div style={{
            background: 'var(--accent-dim)',
            border: '1px solid rgba(0,229,160,0.2)',
            borderRadius: 'var(--r)',
            padding: '10px 12px',
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'var(--accent)',
            lineHeight: 1.7,
          }}>
            <div style={{ fontSize: 10, letterSpacing: '0.1em', marginBottom: 4 }}>★ TOP CONFIG</div>
            chunking: {bestConfig.chunking}<br />
            retrieval: {bestConfig.retrieval}<br />
            rerank: {String(bestConfig.rerank)}<br />
            <span style={{ color: 'rgba(0,229,160,0.7)' }}>
              faith: {(bestConfig.faithfulness * 100).toFixed(1)}% &nbsp;
              rel: {(bestConfig.relevancy * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* API status at bottom */}
      <div style={{ marginTop: 'auto' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)',
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
            background: apiStatus === 'online' ? 'var(--accent)' : apiStatus === 'offline' ? 'var(--danger)' : 'var(--text-muted)',
            boxShadow: apiStatus === 'online' ? '0 0 6px var(--accent-glow)' : 'none',
          }} />
          {apiStatus === 'online' ? 'api online' : apiStatus === 'offline' ? 'api offline' : 'connecting…'}
        </div>
      </div>
    </aside>
  );
}

function ConfigItem({ label, children }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <label style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
        {label}
      </label>
      {children}
    </div>
  );
}

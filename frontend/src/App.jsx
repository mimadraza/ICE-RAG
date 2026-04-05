import React, { useState, useEffect } from 'react';
import QueryPage from './pages/QueryPage';
import { api } from './api/client';
import './index.css';

const DEFAULT_CONFIG = {
  chunking:  'recursive',
  retrieval: 'hybrid',
  rerank:    true,
  retrieveK: 10,
  finalK:    5,
};

export default function App() {
  const [apiStatus, setApiStatus] = useState('connecting');

  useEffect(() => {
    api.health()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const statusColor = apiStatus === 'online' ? '#4CAF50' : apiStatus === 'offline' ? 'var(--accent)' : 'var(--text-muted)';
  const statusLabel = apiStatus === 'online' ? 'SYSTEM ACTIVE' : apiStatus === 'offline' ? 'SYSTEM OFFLINE' : 'CONNECTING...';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>

      {/* Header */}
      <header style={{
        background: 'var(--accent)',
        borderBottom: '5px solid #8B1500',
        padding: '0 32px',
        flexShrink: 0,
      }}>
        {/* Top accent stripe */}
        <div style={{
          height: 4,
          background: 'repeating-linear-gradient(90deg, #8B1500 0px, #8B1500 20px, var(--accent) 20px, var(--accent) 40px)',
          margin: '0 -32px',
          marginBottom: 0,
        }} />

        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 0',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span style={{ fontSize: 28, lineHeight: 1, color: '#F0E8D8' }}>★</span>
            <div>
              <div style={{
                fontFamily: 'var(--sans)',
                fontWeight: 700,
                fontSize: 22,
                color: '#F0E8D8',
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                lineHeight: 1,
              }}>
                NEXUS / KNOWLEDGE ORACLE
              </div>
              <div style={{
                fontFamily: 'var(--mono)',
                fontSize: 10,
                color: 'rgba(240,232,216,0.55)',
                letterSpacing: '0.22em',
                marginTop: 4,
              }}>
                RETRIEVAL-AUGMENTED GENERATION ENGINE
              </div>
            </div>
            <span style={{ fontSize: 28, lineHeight: 1, color: '#F0E8D8' }}>★</span>
          </div>

          {/* Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: 'rgba(240,232,216,0.7)',
            letterSpacing: '0.15em',
          }}>
            <span style={{
              width: 7, height: 7,
              background: statusColor,
              display: 'inline-block',
              boxShadow: apiStatus === 'online' ? `0 0 8px ${statusColor}` : 'none',
            }} />
            {statusLabel}
          </div>
        </div>
      </header>

      {/* Divider bar */}
      <div style={{
        height: 3,
        background: 'repeating-linear-gradient(90deg, var(--border) 0px, var(--border) 8px, transparent 8px, transparent 12px)',
        flexShrink: 0,
      }} />

      {/* Main content */}
      <main style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <QueryPage config={DEFAULT_CONFIG} />
      </main>
    </div>
  );
}

import React from 'react';

/* ── Spinner ── */
export function Spinner({ size = 14 }) {
  return (
    <span style={{
      display: 'inline-block',
      width: size, height: size,
      border: '2px solid rgba(240,232,216,0.2)',
      borderTopColor: '#F0E8D8',
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
      flexShrink: 0,
    }} />
  );
}

/* ── Tag / Chip ── */
export function Tag({ children, variant = 'default' }) {
  const colors = {
    default: { bg: 'var(--surface3)', border: 'var(--border-hi)', color: 'var(--text-muted)' },
    red:     { bg: 'rgba(204,34,0,0.15)',  border: 'rgba(204,34,0,0.4)',  color: 'var(--accent)' },
    amber:   { bg: 'rgba(232,160,0,0.12)', border: 'rgba(232,160,0,0.3)', color: 'var(--accent2)' },
    warn:    { bg: 'rgba(232,160,0,0.12)', border: 'rgba(232,160,0,0.3)', color: 'var(--warn)' },
    danger:  { bg: 'rgba(204,34,0,0.15)',  border: 'rgba(204,34,0,0.4)',  color: 'var(--danger)' },
    /* legacy aliases */
    green:   { bg: 'rgba(204,34,0,0.15)',  border: 'rgba(204,34,0,0.4)',  color: 'var(--accent)' },
    blue:    { bg: 'rgba(232,160,0,0.12)', border: 'rgba(232,160,0,0.3)', color: 'var(--accent2)' },
  };
  const c = colors[variant] || colors.default;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      background: c.bg, border: `1px solid ${c.border}`, color: c.color,
      padding: '2px 7px',
      fontFamily: 'var(--mono)', fontSize: 10, lineHeight: 1.6,
      whiteSpace: 'nowrap',
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
    }}>
      {children}
    </span>
  );
}

/* ── Button ── */
export function Button({ children, variant = 'primary', disabled, onClick, style }) {
  const styles = {
    primary: {
      background: disabled ? 'var(--surface3)' : 'var(--accent)',
      color: disabled ? 'var(--text-muted)' : '#F0E8D8',
      padding: '10px 22px',
      fontFamily: 'var(--sans)',
      fontSize: 14,
      fontWeight: 600,
      letterSpacing: '0.15em',
      textTransform: 'uppercase',
      border: 'none',
      transition: 'background var(--transition), filter var(--transition)',
    },
    ghost: {
      background: 'transparent',
      color: 'var(--text-dim)',
      border: '2px solid var(--border)',
      padding: '8px 16px',
      fontFamily: 'var(--sans)',
      fontSize: 13,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      transition: 'border-color var(--transition), color var(--transition)',
    },
    danger: {
      background: 'rgba(204,34,0,0.15)',
      color: 'var(--danger)',
      border: '2px solid rgba(204,34,0,0.4)',
      padding: '8px 16px',
      fontFamily: 'var(--sans)',
      fontSize: 13,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
    },
  };
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      style={{
        ...styles[variant],
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
        ...style,
      }}
    >
      {children}
    </button>
  );
}

/* ── Card ── */
export function Card({ children, style, animate }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '2px solid var(--border)',
      overflow: 'hidden',
      animation: animate ? 'fadeUp 0.2s ease' : undefined,
      ...style,
    }}>
      {children}
    </div>
  );
}

/* ── Section Label ── */
export function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--mono)',
      fontSize: 9,
      letterSpacing: '0.2em',
      textTransform: 'uppercase',
      color: 'var(--text-muted)',
      marginBottom: 8,
    }}>
      {children}
    </div>
  );
}

/* ── Error box ── */
export function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <div style={{
      background: 'rgba(204,34,0,0.1)',
      border: '2px solid rgba(204,34,0,0.4)',
      borderLeft: '5px solid var(--accent)',
      padding: '12px 16px',
      color: '#F0E8D8',
      fontFamily: 'var(--mono)',
      fontSize: 12,
      lineHeight: 1.6,
      letterSpacing: '0.05em',
    }}>
      <span style={{ color: 'var(--accent)', fontWeight: 500 }}>★ ERROR: </span>{message}
    </div>
  );
}

/* ── Toggle ── */
export function Toggle({ checked, onChange, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      {label && (
        <span style={{
          fontSize: 12,
          color: 'var(--text-dim)',
          fontFamily: 'var(--mono)',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>
          {label}
        </span>
      )}
      <div
        onClick={() => onChange(!checked)}
        style={{
          position: 'relative', width: 36, height: 20, cursor: 'pointer',
          transition: 'background var(--transition)',
          background: checked ? 'rgba(204,34,0,0.2)' : 'var(--surface3)',
          border: checked ? '1px solid var(--accent)' : '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <div style={{
          position: 'absolute', width: 14, height: 14,
          top: 2, left: checked ? 18 : 2,
          transition: 'left var(--transition), background var(--transition)',
          background: checked ? 'var(--accent)' : 'var(--text-muted)',
        }} />
      </div>
    </div>
  );
}

/* ── Select ── */
export function Select({ value, onChange, options, style }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      style={{
        background: 'var(--surface2)',
        border: '1px solid var(--border)',
        color: 'var(--text)',
        fontFamily: 'var(--mono)',
        fontSize: 12,
        padding: '7px 10px',
        width: '100%',
        outline: 'none',
        cursor: 'pointer',
        letterSpacing: '0.05em',
        ...style,
      }}
    >
      {options.map(o => (
        <option key={o.value} value={o.value} style={{ background: 'var(--surface2)' }}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

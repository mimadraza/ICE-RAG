import React, { useState, useRef, useEffect } from 'react';
import { api } from '../api/client';
import AnswerCard from '../components/AnswerCard';
import { Button, Spinner, ErrorBox } from '../components/UI';
import bannerImg from '../assets/banner.png';

export default function QueryPage({ config }) {
  const [query, setQuery]     = useState('');
  const [messages, setMessages] = useState([]); // { type: 'user'|'answer'|'error', content }
  const [loading, setLoading] = useState(false);
  const textareaRef = useRef(null);
  const scrollRef   = useRef(null);

  // Auto-scroll to bottom when new messages arrive or loading changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSubmit = async () => {
    const q = query.trim();
    if (!q || loading) return;

    setMessages(prev => [...prev, { type: 'user', content: q }]);
    setQuery('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setLoading(true);

    try {
      const res = await api.query({
        query: q,
        chunking:   config.chunking,
        retrieval:  config.retrieval,
        rerank:     config.rerank,
        retrieve_k: config.retrieveK,
        final_k:    config.finalK,
      });
      setMessages(prev => [...prev, { type: 'answer', content: res.data }]);
    } catch (e) {
      setMessages(prev => [...prev, { type: 'error', content: e.message }]);
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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>

      {/* Chat area — scrollable, fills available space */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          padding: '24px 32px',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
          backgroundImage: `linear-gradient(rgba(14,12,10,0.80), rgba(14,12,10,0.80)), url(${bannerImg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center 25%',
          backgroundAttachment: 'local',
        }}
      >
        {/* Empty state */}
        {messages.length === 0 && !loading && (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            gap: 16,
            paddingBottom: '60px',
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
              AWAITING QUERY
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

        {/* Message thread */}
        {messages.map((msg, i) => {
          if (msg.type === 'user') {
            return (
              <div key={i} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{
                  maxWidth: '70%',
                  background: 'var(--accent)',
                  color: '#F0E8D8',
                  fontFamily: 'var(--sans)',
                  fontSize: 14,
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  padding: '10px 16px',
                  borderLeft: '4px solid #8B1500',
                  lineHeight: 1.5,
                  wordBreak: 'break-word',
                }}>
                  {msg.content}
                </div>
              </div>
            );
          }
          if (msg.type === 'error') {
            return <ErrorBox key={i} message={msg.content} />;
          }
          return <AnswerCard key={i} result={msg.content} />;
        })}

        {/* Loading indicator */}
        {loading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            color: 'var(--text-muted)',
            fontFamily: 'var(--mono)',
            fontSize: 11,
            letterSpacing: '0.15em',
          }}>
            <Spinner size={13} />
            PROCESSING QUERY...
          </div>
        )}
      </div>

      {/* Input area — pinned to bottom */}
      <div style={{
        padding: '20px 32px',
        borderTop: '2px solid var(--border)',
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
          ▶ QUERY INPUT
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
              placeholder="ENTER YOUR QUERY…  (ENTER TO SEND · SHIFT+ENTER FOR NEW LINE)"
              rows={2}
              style={{
                width: '100%',
                background: 'var(--surface2)',
                border: 'none',
                color: 'var(--accent)',
                fontFamily: 'var(--sans)',
                fontSize: 15,
                fontWeight: 700,
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
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
            {loading ? <Spinner size={13} /> : 'SEARCH'}
          </Button>
        </div>
      </div>

    </div>
  );
}

# RAG Explorer — React Frontend

Vite + React frontend for the RAG pipeline API.

## Setup

```bash
npm install
```

Copy the env example and set your backend URL:

```bash
cp .env.example .env
```

## Dev

```bash
npm run dev   # → http://localhost:5173
```

## Production build

```bash
npm run build        # outputs to dist/
npm run preview      # preview the build locally
```

## Structure

```
src/
├── api/client.js           — All API calls in one place
├── hooks/useApi.js         — Generic fetch hook
├── components/
│   ├── UI.jsx              — Shared primitives (Button, Tag, Card, Toggle, …)
│   ├── Sidebar.jsx         — Nav + live pipeline config + best-config badge
│   └── AnswerCard.jsx      — Query result with collapsible sources + timing
├── pages/
│   ├── QueryPage.jsx       — Query input + answer history
│   ├── ExperimentsPage.jsx — Metrics cards + comparison table
│   └── ChunkingPage.jsx    — Raw chunking report
└── App.jsx                 — Root layout
```

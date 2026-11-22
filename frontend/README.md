# RxCite frontend

React + TypeScript + Vite. One page that sends questions to the FastAPI backend in `../backend`
and shows the cited answers. See "Running the app" in the root README.

```bash
npm install
npm run dev     # http://localhost:5173 (must be this port; the API's CORS allows only it)
npm run build   # type-check and build to dist/
npm run lint    # oxlint
```

Set `VITE_API_URL` in `.env.local` (copy `.env.example`) to use an API other than
http://localhost:8000.

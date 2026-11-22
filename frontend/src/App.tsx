import { useEffect, useState } from "react";
import { API_URL, ApiError, ask, getArchitectures, getHealth } from "./api";
import type { Architecture, ArchitectureId, Health, Mode, Slot } from "./api";
import AnswerCard from "./components/AnswerCard";
import AskForm from "./components/AskForm";

const START_COMMAND = "cd backend && uv run uvicorn api.main:app --reload --port 8000";

function loadBackend(): Promise<[Health, Architecture[]]> {
  return Promise.all([getHealth(), getArchitectures()]);
}

function defaultMode(available: ArchitectureId[]): Mode {
  return available.includes("advanced") ? "advanced" : (available[0] ?? "advanced");
}

function errorMessage(err: unknown, archName: string): string {
  if (!(err instanceof ApiError)) return String(err);
  switch (err.status) {
    case 0:
      return err.message;
    case 422:
      return `Please check your question: ${err.message}`;
    case 503:
      return `${archName} isn't available on this server: ${err.message}`;
    case 502:
      return `An outside service failed: ${err.message}`;
    default:
      return `Error ${err.status}: ${err.message}`;
  }
}

export default function App() {
  const [backend, setBackend] = useState<"checking" | "ok" | "down">("checking");
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [available, setAvailable] = useState<ArchitectureId[]>([]);
  const [unavailable, setUnavailable] = useState<Record<string, string>>({});
  const [missingEnv, setMissingEnv] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("advanced");
  const [asked, setAsked] = useState(""); // the question the shown results answer
  const [results, setResults] = useState<Partial<Record<ArchitectureId, Slot>>>({});

  const loading = Object.values(results).some((s) => s?.status === "loading");

  function applyBackend([health, archs]: [Health, Architecture[]]) {
    setArchitectures(archs);
    setAvailable(health.architectures);
    setUnavailable(health.unavailable);
    setMissingEnv(health.missing_env);
    setMode((m) =>
      (m === "all" && health.architectures.length > 0) || health.architectures.includes(m as ArchitectureId)
        ? m
        : defaultMode(health.architectures),
    );
    setBackend("ok");
  }

  // A timeout doesn't mean the server is down, so ask /health before showing the banner
  function checkBackend() {
    loadBackend().then(applyBackend, () => setBackend("down"));
  }

  useEffect(() => {
    let ignore = false; // StrictMode runs effects twice in dev
    loadBackend().then(
      (r) => !ignore && applyBackend(r),
      () => !ignore && setBackend("down"),
    );
    return () => {
      ignore = true;
    };
  }, []);

  function retry() {
    setBackend("checking");
    checkBackend();
  }

  const targets: ArchitectureId[] =
    mode === "all" ? architectures.map((a) => a.id).filter((id) => available.includes(id)) : [mode];

  function submit() {
    const q = question.trim();
    if (!q || loading || targets.length === 0) return;
    setAsked(q);
    setResults(Object.fromEntries(targets.map((id) => [id, { status: "loading" }])));
    for (const id of targets) {
      const name = architectures.find((a) => a.id === id)?.name ?? id;
      ask(q, id)
        .then((data): Slot => ({ status: "done", data }))
        .catch((err: unknown): Slot => {
          if (err instanceof ApiError && err.status === 0) checkBackend();
          return { status: "error", message: errorMessage(err, name) };
        })
        // functional update so parallel responses don't overwrite each other
        .then((slot) => setResults((prev) => ({ ...prev, [id]: slot })));
    }
  }

  const shown = architectures.filter((a) => results[a.id]);
  const compare = shown.length > 1;

  return (
    <main className={compare ? "wide" : undefined}>
      <h1>RxCite: cited answers from FDA drug labels</h1>
      <p className="disclaimer">
        RxCite is a class project, not medical advice. Answers come only from FDA drug labels and
        may be incomplete or out of date. Talk to a pharmacist or doctor before starting, stopping
        or changing any medicine. In an emergency, call 911.
      </p>

      {backend === "checking" && <p className="muted">Connecting to the server…</p>}
      {backend === "down" && (
        <div className="banner" role="alert">
          <p>
            Can't reach the RxCite server at {API_URL}. Is the backend running? Start it with:
          </p>
          <pre>
            <code>{START_COMMAND}</code>
          </pre>
          <button type="button" onClick={retry}>
            Retry
          </button>
        </div>
      )}
      {backend === "ok" && missingEnv.length > 0 && (
        <p className="banner">
          The server is missing settings: {missingEnv.join(", ")} (see backend/.env). Questions
          will fail until they're set.
        </p>
      )}

      <AskForm
        question={question}
        setQuestion={setQuestion}
        mode={mode}
        setMode={setMode}
        architectures={architectures}
        available={available}
        unavailable={unavailable}
        loading={loading}
        canAsk={backend === "ok" && targets.length > 0}
        onSubmit={submit}
      />

      <div aria-live="polite">
        {asked && shown.length > 0 && <p>Answers for: {asked}</p>}
        <div className={compare ? "compare" : undefined}>
          {shown.map((a) => (
            <AnswerCard key={a.id} title={a.name} slot={results[a.id]!} />
          ))}
        </div>
      </div>
    </main>
  );
}

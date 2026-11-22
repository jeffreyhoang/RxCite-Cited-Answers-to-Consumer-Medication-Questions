import type { FormEvent } from "react";
import type { Architecture, ArchitectureId, Mode } from "../api";

const EXAMPLES = [
  "Can I drink alcohol with ibuprofen?",
  "Should I stop taking my lisinopril?",
  "What is the dose of metformin and what are its side effects?",
];

interface Props {
  question: string;
  setQuestion: (q: string) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
  architectures: Architecture[];
  available: ArchitectureId[];
  unavailable: Record<string, string>;
  loading: boolean;
  canAsk: boolean;
  onSubmit: () => void;
}

export default function AskForm(props: Props) {
  const { question, setQuestion, mode, setMode, architectures, available, unavailable } = props;
  const { loading, canAsk, onSubmit } = props;

  function submit(e: FormEvent) {
    e.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={submit}>
      <fieldset disabled={loading}>
        <label htmlFor="question">Your question</label>
        <textarea
          id="question"
          rows={3}
          maxLength={1000}
          placeholder="e.g. Can I drink alcohol with ibuprofen?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <p className="examples">
          Examples:{" "}
          {EXAMPLES.map((q) => (
            <button key={q} type="button" onClick={() => setQuestion(q)}>
              {q}
            </button>
          ))}
        </p>
        <div className="modes" role="radiogroup" aria-label="Architecture">
          {architectures.map((a) => {
            const ok = available.includes(a.id);
            return (
              <label key={a.id} title={ok ? a.description : unavailable[a.id] ?? a.description}>
                <input
                  type="radio"
                  name="mode"
                  value={a.id}
                  checked={mode === a.id}
                  disabled={!ok}
                  onChange={() => setMode(a.id)}
                />
                {a.name}
                {!ok && " (unavailable)"}
              </label>
            );
          })}
          <label title="Ask every available architecture at once and compare the answers">
            <input
              type="radio"
              name="mode"
              value="all"
              checked={mode === "all"}
              disabled={available.length === 0}
              onChange={() => setMode("all")}
            />
            Compare all four
          </label>
        </div>
      </fieldset>
      <button type="submit" disabled={loading || !canAsk || question.trim() === ""}>
        {loading ? "Asking…" : "Ask"}
      </button>
    </form>
  );
}

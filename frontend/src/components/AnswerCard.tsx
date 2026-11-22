import type { AskResponse, Slot } from "../api";
import { safeUrl } from "../api";
import CitedText from "./CitedText";
import SourceList from "./SourceList";

// info is untyped on the server, so check value types
function strings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function infoLine({ info, latency_s }: AskResponse): string {
  const parts: string[] = [];
  if (typeof info.search_query === "string") parts.push(`Search query: "${info.search_query}"`);
  const subs = strings(info.sub_questions);
  if (subs.length) parts.push(`Sub-questions: ${subs.join("; ")}`);
  const drugs = strings(info.drugs);
  if (drugs.length) parts.push(`Drugs: ${drugs.join(", ")}`);
  if (typeof info.section_type === "string") parts.push(`Section: ${info.section_type}`);
  parts.push(`${latency_s.toFixed(1)} s`);
  return parts.join(" · ");
}

function Answer({ data }: { data: AskResponse }) {
  if (data.refused) {
    return (
      <>
        <p className="muted">{infoLine(data)}</p>
        <div className="refusal">
          <strong>Not answered from the labels</strong>
          <p className="answer">{data.answer}</p>
        </div>
        <SourceList sources={data.sources} />
      </>
    );
  }
  return (
    <>
      <p className="muted">{infoLine(data)}</p>
      <CitedText text={data.answer} citations={data.citations} />
      {data.citations.length > 0 && (
        <>
          <h3>Citations</h3>
          <ol className="citations">
            {data.citations.map((c) => {
              const href = safeUrl(c.url);
              return (
                <li key={c.n}>
                  [{c.n}]{" "}
                  {href ? (
                    <a href={href} target="_blank" rel="noopener noreferrer">
                      {c.label}
                    </a>
                  ) : (
                    c.label
                  )}
                </li>
              );
            })}
          </ol>
        </>
      )}
      <SourceList sources={data.sources} />
    </>
  );
}

export default function AnswerCard({ title, slot }: { title: string; slot: Slot }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {slot.status === "loading" && (
        <p className="muted">Looking through drug labels… (usually 3–8 s)</p>
      )}
      {slot.status === "error" && (
        <div className="error">
          <strong>Error:</strong> {slot.message}
        </div>
      )}
      {slot.status === "done" && <Answer data={slot.data} />}
    </section>
  );
}

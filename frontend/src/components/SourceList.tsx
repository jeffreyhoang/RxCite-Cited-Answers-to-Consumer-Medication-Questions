import type { Source } from "../api";
import { safeUrl } from "../api";

export default function SourceList({ sources }: { sources: Source[] }) {
  return (
    <details>
      <summary>Retrieved sources ({sources.length})</summary>
      <ol className="sources">
        {sources.map((s) => {
          const href = safeUrl(s.url);
          return (
            <li key={s.chunk_id}>
              <details>
                <summary>
                  #{s.rank} · {s.drug_name} — {s.section_label} · score {s.score.toFixed(2)}
                  {s.cited && (
                    <>
                      {" · "}
                      <strong>cited</strong>
                    </>
                  )}
                </summary>
                {href && (
                  <p>
                    <a href={href} target="_blank" rel="noopener noreferrer">
                      View on DailyMed
                    </a>
                  </p>
                )}
                <div className="source-text">{s.text}</div>
              </details>
            </li>
          );
        })}
      </ol>
    </details>
  );
}

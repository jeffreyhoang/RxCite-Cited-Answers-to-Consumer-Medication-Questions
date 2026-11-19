import { Fragment } from "react";
import { safeUrl } from "../api";
import type { Citation } from "../api";

// Text parts render as plain strings (escaped by React), never as HTML
export default function CitedText({ text, citations }: { text: string; citations: Citation[] }) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  // even indexes: text; odd indexes: marker numbers
  const parts = text.split(/\[(\d+)\]/);
  return (
    <p className="answer">
      {parts.map((part, i) => {
        if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
        const c = byN.get(Number(part));
        const href = c && safeUrl(c.url);
        return (
          <Fragment key={i}>
            {c && href ? (
              <a href={href} target="_blank" rel="noopener noreferrer" title={c.label}>
                [{part}]
              </a>
            ) : (
              `[${part}]` // unknown marker stays text
            )}
          </Fragment>
        );
      })}
    </p>
  );
}

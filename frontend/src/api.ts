export type ArchitectureId = "naive" | "advanced" | "hierarchical" | "decomposition";
export type Mode = ArchitectureId | "all";

export interface Architecture {
  id: ArchitectureId;
  name: string;
  description: string;
}

export interface Health {
  status: "ok" | "degraded";
  architectures: ArchitectureId[]; // only the ones that loaded
  unavailable: Record<string, string>; // id -> load error
  missing_env: string[];
}

export interface Citation {
  n: number;
  set_id: string;
  section: string;
  section_label: string;
  drug_name: string;
  label: string;
  url: string;
}

export interface Source {
  rank: number;
  chunk_id: string;
  set_id: string;
  drug_name: string;
  section: string;
  section_label: string;
  text: string;
  score: number;
  url: string;
  cited: boolean;
}

// Keys depend on the architecture
export interface AskInfo {
  search_query?: string;
  rerank_s?: number;
  sub_questions?: string[];
  drugs?: string[];
  section_type?: string;
}

export interface AskResponse {
  question: string;
  architecture: ArchitectureId;
  answer: string;
  refused: boolean;
  citations: Citation[];
  sources: Source[];
  info: AskInfo;
  latency_s: number;
}

export type Slot =
  | { status: "loading" }
  | { status: "done"; data: AskResponse }
  | { status: "error"; message: string };

export class ApiError extends Error {
  status: number; // 0 = no response
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// https only, so bad data can't produce a javascript: link
export function safeUrl(url: string): string | undefined {
  return url.startsWith("https://") ? url : undefined;
}

export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const TIMEOUT_MS = 60_000;

// FastAPI sends {"detail": string}, or a list of {msg} for a 422
function errorDetail(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => String((d as { msg?: unknown })?.msg ?? d)).join("; ");
  }
  return `Request failed (HTTP ${status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_URL + path, { ...init, signal: AbortSignal.timeout(TIMEOUT_MS) });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new ApiError(0, `The server took too long to answer (over ${TIMEOUT_MS / 1000} s).`);
    }
    throw new ApiError(0, `Can't reach the RxCite server at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // not JSON
    }
    throw new ApiError(res.status, errorDetail(body, res.status));
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function getArchitectures(): Promise<Architecture[]> {
  return request<Architecture[]>("/architectures");
}

export function ask(question: string, architecture: ArchitectureId): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question.trim(), architecture }),
  });
}

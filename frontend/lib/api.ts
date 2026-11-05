"use client";

/**
 * API client. Authentication is an HttpOnly session cookie set by the API (JavaScript cannot read it).
 * State-changing requests carry the CSRF token, which is kept in memory only, never in storage.
 */
// An empty NEXT_PUBLIC_API_URL means "same origin": the API is served behind the same host as this app
// (single-container deployment), so requests stay first-party and the SameSite=Strict session cookie is sent.
export const API = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

function wsBase(): string {
  if (API) return API.replace(/^http/, "ws");
  const { protocol, host } = window.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}`;
}

let csrf: string | null = null;
export function setCsrf(t: string | null) { csrf = t; }

export interface Me {
  username: string; display_name: string; roles: string[]; permissions: string[]; must_change_password: boolean;
}
export interface AuthState { setup_required: boolean; signed_in: boolean; user?: Me; csrf_token?: string }

function headers(json: boolean, unsafe: boolean): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (unsafe && csrf) h["X-CSRF-Token"] = csrf;
  return h;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

function onAuthLost() {
  if (typeof window === "undefined") return;
  const p = window.location.pathname;
  if (!p.startsWith("/login") && !p.startsWith("/setup")) {
    window.location.href = `/login?next=${encodeURIComponent(p + window.location.search)}`;
  }
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : "Request was rejected"; } catch {}
    if (r.status === 401) onAuthLost();
    if (r.status === 403 && msg === "Password change required" && typeof window !== "undefined" && !window.location.pathname.startsWith("/account")) {
      window.location.href = "/account?required=1";
    }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

const opts = (method: string, body?: BodyInit, json = true): RequestInit => ({
  method, body, credentials: "include", cache: "no-store", headers: headers(json && !!body, method !== "GET"),
});

export const api = {
  get: <T,>(path: string) => fetch(`${API}${path}`, opts("GET")).then(handle<T>),
  post: <T,>(path: string, body?: unknown) =>
    fetch(`${API}${path}`, opts("POST", body ? JSON.stringify(body) : undefined)).then(handle<T>),
  patch: <T,>(path: string, body: unknown) => fetch(`${API}${path}`, opts("PATCH", JSON.stringify(body))).then(handle<T>),
  upload: <T,>(path: string, file: File, kind: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("kind", kind);
    return fetch(`${API}${path}`, opts("POST", fd, false)).then(handle<T>);
  },
  audioUrl: async (recId: string) => {
    const r = await fetch(`${API}/api/recordings/${recId}/audio`, opts("GET"));
    if (!r.ok) throw new ApiError(r.status, r.status === 403 ? "You are not allowed to open this recording" : "Could not load evidence audio");
    return URL.createObjectURL(await r.blob());
  },
  /** The session cookie is sent with the WebSocket handshake; nothing secret goes in the URL. */
  wsUrl: (caseId: string) => `${wsBase()}/ws/cases/${encodeURIComponent(caseId)}/live`,
};

export const authApi = {
  state: async (): Promise<AuthState> => {
    const s = await fetch(`${API}/api/auth/state`, opts("GET")).then(handle<AuthState>);
    setCsrf(s.csrf_token || null);
    return s;
  },
  login: async (username: string, password: string) => {
    const r = await fetch(`${API}/api/auth/login`, opts("POST", JSON.stringify({ username, password }))).then(handle<{ user: Me; csrf_token: string }>);
    setCsrf(r.csrf_token);
    return r;
  },
  setup: async (body: { username: string; display_name: string; password: string }) => {
    const r = await fetch(`${API}/api/auth/setup`, opts("POST", JSON.stringify(body))).then(handle<{ user: Me; csrf_token: string }>);
    setCsrf(r.csrf_token);
    return r;
  },
  logout: async () => {
    try { await api.post("/api/auth/logout"); } finally { setCsrf(null); }
  },
};

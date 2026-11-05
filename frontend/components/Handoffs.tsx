"use client";

import clsx from "clsx";
import { PhoneForwarded, PhoneOff, ShieldAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { Handoff } from "@/lib/types";

const OPEN = ["pending", "connecting", "suggested"];

/** Helpline hand-offs for one case: the automatic countdown, one-click suggestions and what happened. */
export function Handoffs({ caseId, initial }: { caseId: string; initial?: Handoff[] | null }) {
  const [items, setItems] = useState<Handoff[]>(initial || []);
  const [now, setNow] = useState(Date.now());
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (initial) setItems(initial); }, [initial]);

  const active = useMemo(() => items.some((h) => h.status === "pending" || h.status === "connecting"), [items]);
  useEffect(() => {
    if (!active) return;
    const tick = setInterval(() => setNow(Date.now()), 250);
    const poll = setInterval(async () => {
      try { setItems(await api.get<Handoff[]>(`/api/cases/${caseId}/handoffs`)); } catch { /* keep the last state */ }
    }, 1000);
    return () => { clearInterval(tick); clearInterval(poll); };
  }, [active, caseId]);

  if (!items.length) return null;

  const act = async (h: Handoff, what: "connect" | "cancel") => {
    setBusy(h.id); setErr(null);
    try {
      const r = await api.post<Handoff>(`/api/cases/${caseId}/handoffs/${h.id}/${what}`,
        what === "cancel" ? { reason: "Cancelled by the officer" } : {});
      setItems((xs) => xs.map((x) => (x.id === r.id ? r : x)));
    } catch (e: any) { setErr(e.message); }
    setBusy(null);
  };

  const ordered = [...items].sort((a, b) => Number(OPEN.includes(b.status)) - Number(OPEN.includes(a.status)));
  return (
    <section className="panel" aria-live="assertive">
      <header className="panel-h">
        <h3 className="panel-t flex items-center gap-1.5"><PhoneForwarded size={14} /> Helpline hand-off</h3>
        <span className="text-xs2 text-fg-3">{items[0]?.mode === "demo" ? "Demo mode: no real call is placed" : "Warm three-way transfer"}</span>
      </header>
      <ul className="divide-y divide-line">
        {ordered.map((h) => {
          const left = h.execute_at ? Math.max(0, Math.ceil((Date.parse(h.execute_at) - now) / 1000)) : 0;
          const tone = h.status === "pending" || h.status === "connecting" ? "crit"
            : h.status === "suggested" ? "warn" : h.status === "connected" ? "ok" : "muted";
          return (
            <li key={h.id} className={clsx("px-4 py-3 text-[13px]",
              tone === "crit" && "bg-crit-soft", tone === "warn" && "bg-warn-soft/40")}>
              <div className="flex flex-wrap items-center gap-2">
                {tone === "crit" ? <ShieldAlert size={15} className="text-crit" /> : null}
                <span className="font-medium">{h.name} · {h.number}</span>
                <span className="text-fg-3">
                  {h.status === "pending" && (left > 0 ? `connecting in ${left} s` : "connecting…")}
                  {h.status === "connecting" && "connecting…"}
                  {h.status === "connected" && "joined the call (officer stays on until they take over)"}
                  {h.status === "suggested" && "suggested"}
                  {h.status === "cancelled" && `cancelled${h.actor ? ` by ${h.actor}` : ""}`}
                  {h.status === "dismissed" && "dismissed"}
                  {h.status === "failed" && "not connected: dial manually"}
                </span>
                <span className="ml-auto flex gap-2">
                  {(h.status === "pending" || h.status === "suggested" || h.status === "failed") && (
                    <button className="btn btn-xs border-brand bg-brand text-white" disabled={busy === h.id}
                      onClick={() => act(h, "connect")}>
                      <PhoneForwarded size={12} /> {h.status === "pending" ? "Connect now" : "Transfer"}
                    </button>
                  )}
                  {(h.status === "pending" || h.status === "suggested") && (
                    <button className="btn-secondary btn-xs" disabled={busy === h.id} onClick={() => act(h, "cancel")}>
                      <PhoneOff size={12} /> {h.status === "pending" ? "Cancel" : "Dismiss"}
                    </button>
                  )}
                </span>
              </div>
              <div className="mt-1 text-xs2 text-fg-2">{h.reason}</div>
              {h.detail && h.status !== "pending" && <div className="mt-0.5 text-xs2 text-fg-3">{h.detail}</div>}
            </li>
          );
        })}
      </ul>
      {err && <div className="px-4 pb-3 text-xs2 text-crit">{err}</div>}
      <p className="border-t border-line px-4 py-2 text-xs2 text-fg-3">
        Words-based signal, indicative only. An automatic transfer can be cancelled during the countdown; every step is in the audit log.
      </p>
    </section>
  );
}

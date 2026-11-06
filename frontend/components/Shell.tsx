"use client";

import clsx from "clsx";
import { BarChart3, FilePlus2, Inbox, LogOut, ScrollText, Share2, UserCog, Users, FlaskConical } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, authApi } from "@/lib/api";
import { OPEN_PATHS, PUBLIC_PATHS, SessionProvider, useSession } from "@/lib/session";
import { IncomingCall } from "@/components/IncomingCall";
import type { Health } from "@/lib/types";
import { useWorkspace, workspaceChosen } from "@/lib/workspace";

export function Logo({ height = 22 }: { height?: number }) {
  // eslint-disable-next-line @next/next/no-img-element
  return <img src="/brand/anvesh-wordmark.png" alt="Anvesh" style={{ height }} className="w-auto select-none" draggable={false} />;
}

const NAV = [
  { href: "/", label: "Case queue", icon: Inbox, perm: "case:read", match: (p: string) => p === "/" || p.startsWith("/cases") },
  { href: "/intake", label: "New intake", icon: FilePlus2, perm: "case:write", match: (p: string) => p.startsWith("/intake") },
  { href: "/rings", label: "Linked cases", icon: Share2, perm: "case:read", match: (p: string) => p.startsWith("/rings") },
  { href: "/evidence", label: "Evidence", icon: FlaskConical, perm: "", match: (p: string) => p.startsWith("/evidence") },
  { href: "/insights", label: "National data", icon: BarChart3, perm: "", match: (p: string) => p.startsWith("/insights") },
  { href: "/admin/users", label: "Users", icon: Users, perm: "user:admin", match: (p: string) => p.startsWith("/admin") },
  { href: "/audit", label: "Audit log", perm: "audit:read", icon: ScrollText, match: (p: string) => p.startsWith("/audit") },
];

export function Shell({ children }: { children: React.ReactNode }) {
  return <SessionProvider><Frame>{children}</Frame></SessionProvider>;
}

const ROLE_LABEL: Record<string, string> = { operator: "Operator", supervisor: "Supervisor", admin: "Administrator" };

function Frame({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const { me, ready, can } = useSession();
  if ([...PUBLIC_PATHS, ...OPEN_PATHS].some((p) => path.startsWith(p))) return <>{children}</>;
  if (!ready || !me) return <div className="min-h-screen bg-bg" />;
  if (path.startsWith("/call")) return <>{children}</>;  // the caller's phone: no officer chrome
  return <Chrome can={can} me={me}>{can("case:write") && <IncomingCall />}{children}</Chrome>;
}

function Chrome({ children, can, me }: { children: React.ReactNode; can: (p: string) => boolean; me: NonNullable<ReturnType<typeof useSession>["me"]> }) {
  const path = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  const [ws, setWs] = useWorkspace();
  // Until someone takes a real call, open on the training data so the dashboards are not empty.
  useEffect(() => {
    if (!can("case:read") || workspaceChosen()) return;
    api.get<{ open_cases: number; training_available?: number }>("/api/stats").then((s) => {
      if (s.open_cases === 0 && (s.training_available ?? 0) > 0) setWs("training", false);
      else if (s.open_cases > 0) setWs("live", false);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = () => api.get<Health>("/api/health").then((h) => { if (alive) { setHealth(h); setDown(false); } })
      .catch(() => alive && setDown(true));
    tick();
    const t = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const models = health?.models || [];
  const m = (n: string) => models.find((x) => x.name === n);

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-[216px] shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <Link href="/" className="flex h-14 items-center border-b border-line px-4" aria-label="Anvesh home">
          <Logo height={21} />
        </Link>
        <nav className="flex-1 space-y-0.5 p-2">
          {NAV.filter((n) => !n.perm || can(n.perm)).map(({ href, label, icon: Icon, match }) => (
            <Link key={href} href={href}
              className={clsx("relative flex h-8 items-center gap-2.5 rounded px-2.5 text-[13px] font-medium",
                match(path) ? "bg-brand-soft text-fg" : "text-fg-2 hover:bg-bg hover:text-fg")}>
              {match(path) && <span className="absolute -left-2 top-1.5 h-5 w-[3px] rounded-r bg-brand" />}
              <Icon size={15} strokeWidth={1.8} className={match(path) ? "text-brand-700" : "text-fg-3"} />
              {label}
            </Link>
          ))}
        </nav>
        {can("case:read") && <div className="px-4 pb-3">
          <div className="mb-1.5 text-xs2 font-medium text-fg-3">Workspace</div>
          <div className="grid grid-cols-2 rounded border border-line2 p-0.5 text-xs2 font-medium">
            {(["live", "training"] as const).map((w) => (
              <button key={w} onClick={() => setWs(w)}
                className={clsx("rounded-[4px] py-1", ws === w ? (w === "live" ? "bg-brand text-white" : "bg-warn-soft text-warn") : "text-fg-3 hover:text-fg")}>
                {w === "live" ? "Live cases" : "Training"}
              </button>
            ))}
          </div>
        </div>}
        <div className="flex items-center gap-2 border-t border-line px-4 py-3">
          <Link href="/account" className="min-w-0 flex-1 rounded hover:bg-bg" title="Account and password">
            <div className="truncate text-[13px] font-medium text-fg">{me.display_name}</div>
            <div className="truncate text-xs2 text-fg-3">{me.roles.map((r) => ROLE_LABEL[r] || r).join(" · ")}</div>
          </Link>
          <Link href="/account" className="btn-quiet btn-xs px-1.5" aria-label="Account"><UserCog size={15} /></Link>
          <button onClick={async () => { await authApi.logout(); window.location.href = "/login"; }}
            className="btn-quiet btn-xs px-1.5" aria-label="Sign out" title="Sign out"><LogOut size={15} /></button>
        </div>
        <div className="border-t border-line p-4 text-xs2">
          <div className="mb-2 font-medium text-fg-3">System</div>
          <Status label="API" state={down ? "off" : health ? "on" : "wait"} text={down ? "Unreachable" : health ? "Online" : "…"} />
          <ModelStatus label="Speech-to-text (live)" m={m("asr")} down={down} />
          <ModelStatus label="Speech-to-text (accurate)" m={m("asr_final")} down={down} />
          <ModelStatus label="Voice emotion" m={m("voice_emotion")} down={down} />
          <ModelStatus label="Synthetic-voice check" m={m("voice_authenticity")} down={down} />
        </div>
      </aside>
      <main className="min-w-0 flex-1">
        {down && (
          <div className="border-b border-crit-line bg-crit-soft px-6 py-2 text-[13px] text-crit">
            Cannot reach the Anvesh API. Start the backend (start.bat) and reload.
          </div>
        )}
        {children}
      </main>
    </div>
  );
}

const STATE_TEXT: Record<string, string> = { ready: "Ready", loading: "Loading…", idle: "Not loaded", error: "Error", disabled: "Off" };

function ModelStatus({ label, m, down }: { label: string; m?: Health["models"][number]; down: boolean }) {
  const st = down ? "off" : m?.state || (m?.loaded ? "ready" : "idle");
  const dot = st === "ready" ? "on" : st === "error" || st === "off" ? "off" : "idle";
  return <span title={m?.error || ""}><Status label={label} state={dot} text={down ? "—" : STATE_TEXT[st] || st} /></span>;
}

function Status({ label, state, text }: { label: string; state: "on" | "off" | "idle" | "wait"; text: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-fg-2">{label}</span>
      <span className="flex items-center gap-1.5 text-fg-3">
        <span className={clsx("h-1.5 w-1.5 rounded-full", { on: "bg-ok", off: "bg-crit", idle: "bg-fg-4", wait: "bg-fg-4" }[state])} />
        {text}
      </span>
    </div>
  );
}

export function TopBar({ title, crumbs, right }: { title: React.ReactNode; crumbs?: { href: string; label: string }[]; right?: React.ReactNode }) {
  return (
    <div className="sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b border-line bg-surface/95 px-6 backdrop-blur">
      <div className="flex min-w-0 items-center gap-2 text-[13px]">
        {crumbs?.map((c) => (
          <span key={c.href} className="flex items-center gap-2 text-fg-3">
            <Link href={c.href} className="hover:text-fg">{c.label}</Link><span>/</span>
          </span>
        ))}
        <h1 className="truncate text-[15px] font-semibold text-fg">{title}</h1>
      </div>
      <div className="flex shrink-0 items-center gap-2">{right}</div>
    </div>
  );
}

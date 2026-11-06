"use client";

import { PhoneIncoming } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

/** Rings on every officer console when a caller dials in from the /call page. */
export function IncomingCall() {
  const router = useRouter();
  const [calls, setCalls] = useState<{ id: string; number: string; seconds: number }[]>([]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const ctx = useRef<AudioContext | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => api.get<typeof calls>("/api/calls/incoming").then((x) => alive && setCalls(x)).catch(() => {});
    tick();
    const t = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const showing = calls.filter((c) => !hidden.has(c.id));
  useEffect(() => {
    if (!showing.length) return;
    const ring = () => {
      try {
        ctx.current = ctx.current || new AudioContext();
        const a = ctx.current, o = a.createOscillator(), g = a.createGain();
        o.frequency.value = 660; g.gain.value = 0.05; o.connect(g); g.connect(a.destination);
        o.start(); setTimeout(() => o.stop(), 350);
      } catch {}
    };
    ring();
    const t = setInterval(ring, 1500);
    return () => clearInterval(t);
  }, [showing.length]);

  if (!showing.length) return null;
  const c = showing[0];
  const answer = async () => {
    try { await api.post(`/api/calls/${c.id}/answer`); } catch {}
    setHidden((h) => new Set(h).add(c.id));
    router.push(`/cases/${c.id}?answered=1`);
  };
  return (
    <div className="fixed left-1/2 top-3 z-50 flex -translate-x-1/2 items-center gap-3 rounded-full border border-ok-line bg-surface py-1.5 pl-2 pr-1.5 shadow-xl no-print">
      <span className="flex h-8 w-8 animate-pulse items-center justify-center rounded-full bg-ok text-white"><PhoneIncoming size={16} /></span>
      <div className="text-[13px]"><b>Incoming call</b> · {c.number} · <span className="text-fg-3">{c.seconds} s</span></div>
      <button onClick={answer} className="btn btn-xs rounded-full border-ok bg-ok px-3 text-white">Answer</button>
      <button onClick={() => setHidden((h) => new Set(h).add(c.id))} className="btn-quiet btn-xs rounded-full">Later</button>
    </div>
  );
}

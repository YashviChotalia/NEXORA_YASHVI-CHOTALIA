"use client";

import clsx from "clsx";
import { Mic, Phone, PhoneOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Logo } from "@/components/Shell";
import { api } from "@/lib/api";
import { startLive, type LiveHandle } from "@/lib/live";

/** The caller's side of a helpline call, for demos and training: this browser is the caller's phone.
 *  Its microphone streams over the same live connection an officer's microphone uses, so everything the
 *  officer console shows comes from the real pipeline. Open it on a second screen or a phone on the same network. */

const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"];

export default function CallerPhone() {
  const [number, setNumber] = useState("1930");
  const [state, setState] = useState<"idle" | "ringing" | "connected" | "ended">("idle");
  const [err, setErr] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [secs, setSecs] = useState(0);
  const [caseId, setCaseId] = useState<string | null>(null);
  const live = useRef<LiveHandle | null>(null);

  useEffect(() => {
    if (state !== "connected" && state !== "ringing") return;
    const t = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [state]);
  useEffect(() => () => live.current?.stop(), []);

  const dial = async () => {
    setErr(null); setSecs(0); setState("ringing");
    try {
      const r = await api.post<{ id: string }>("/api/calls", { number });
      setCaseId(r.id);
      live.current = await startLive(api.wsUrl(r.id), (msg) => {
        if (msg.type === "update" || msg.type === "emotion") setState("connected");
        if (msg.type === "error") setErr(msg.message);
      }, setLevel, (m) => { setErr(m); setState("ended"); live.current?.stop(); });
      setTimeout(() => setState((s) => (s === "ringing" ? "connected" : s)), 2500);
    } catch (e: any) { setErr(e.message); setState("idle"); }
  };
  const hang = () => { live.current?.stop(); live.current = null; setLevel(0); setState("ended"); };
  const mmss = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0B0F17] p-4">
      <div className="w-full max-w-[340px] rounded-[36px] border border-white/10 bg-[#111827] px-6 pb-8 pt-6 text-white shadow-2xl">
        <div className="flex items-center justify-between text-[11px] text-white/50">
          <span>Caller phone (demo)</span><span className="opacity-80"><Logo height={12} /></span>
        </div>
        <div className="mt-10 text-center">
          <div className="text-[11px] uppercase tracking-[0.18em] text-white/40">
            {state === "idle" ? "Enter number" : state === "ringing" ? "Calling…" : state === "connected" ? "Connected · cyber cell" : "Call ended"}
          </div>
          <div className="mt-2 font-mono text-[40px] tracking-wider">{number || " "}</div>
          <div className="mt-1 h-5 text-[13px] text-white/60">{state === "connected" || state === "ringing" ? mmss : number === "1930" ? "National cyber crime helpline" : ""}</div>
        </div>

        {state === "idle" || state === "ended" ? (
          <div className="mt-8 grid grid-cols-3 gap-4">
            {KEYS.map((k) => (
              <button key={k} onClick={() => setNumber((n) => (n.length < 12 ? n + k : n))}
                className="mx-auto h-16 w-16 rounded-full bg-white/10 text-[24px] font-light hover:bg-white/20">{k}</button>
            ))}
          </div>
        ) : (
          <div className="mt-10 flex flex-col items-center gap-3">
            <div className="relative flex h-28 w-28 items-center justify-center">
              <span className="absolute inset-0 rounded-full bg-emerald-500/20" style={{ transform: `scale(${1 + Math.min(1, level * 3) * 0.35})`, transition: "transform 120ms" }} />
              <span className="relative flex h-20 w-20 items-center justify-center rounded-full bg-emerald-600"><Mic size={30} /></span>
            </div>
            <p className="max-w-[240px] text-center text-[12px] leading-[17px] text-white/55">
              Speak as the caller. The officer console hears this microphone live.
            </p>
            {caseId && <p className="font-mono text-[11px] text-white/35">{caseId}</p>}
          </div>
        )}

        {err && <p className="mt-4 rounded-lg bg-red-500/15 px-3 py-2 text-[12px] text-red-200">{err}</p>}

        <div className="mt-8 flex justify-center gap-6">
          {state === "idle" || state === "ended" ? (
            <>
              <button onClick={() => setNumber((n) => n.slice(0, -1))} className="h-16 w-16 rounded-full text-[13px] text-white/60 hover:text-white">Delete</button>
              <button onClick={dial} disabled={!number} aria-label="Call"
                className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40"><Phone size={26} /></button>
              <span className="h-16 w-16" />
            </>
          ) : (
            <button onClick={hang} aria-label="Hang up"
              className={clsx("flex h-16 w-16 items-center justify-center rounded-full bg-red-600 hover:bg-red-500")}><PhoneOff size={26} /></button>
          )}
        </div>
      </div>
    </div>
  );
}

"use client";

import clsx from "clsx";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { TopBar } from "@/components/Shell";
import { ErrorNote } from "@/components/ui";
import { api } from "@/lib/api";
import { LANGUAGES } from "@/lib/format";
import { useWorkspace } from "@/lib/workspace";
import type { CaseDetail, Mode } from "@/lib/types";

const MODES: { key: Mode; title: string; desc: string; channel: string; runs: string; off: string }[] = [
  { key: "victim", title: "Victim report", channel: "helpline",
    desc: "A complainant reporting a financial cyber fraud.",
    runs: "Detail extraction, triage, interview prompts, read-back, distress and coercion alerts",
    off: "Scam-tactic scoring of the caller's own words" },
  { key: "suspect_recording", title: "Suspected scam recording", channel: "upload",
    desc: "Audio of the suspected fraudster, shared by a victim.",
    runs: "Scam-tactic tagging, identifier extraction, case linking including voice leads",
    off: "Emotion and voice-stress analysis of the suspect" },
  { key: "informant", title: "Informant tip", channel: "informant_line",
    desc: "Information about fraud activity from a third party.",
    runs: "Lead extraction, case linking, confidentiality prompts",
    off: "Using emotion as a measure of credibility" },
];

const SOURCES = [
  { key: "live", label: "Live microphone" },
  { key: "upload", label: "Audio file" },
  { key: "text", label: "Typed transcript" },
] as const;

export default function Intake() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("victim");
  const [caller, setCaller] = useState("");
  const [district, setDistrict] = useState("");
  const [language, setLanguage] = useState("auto");
  const [start, setStart] = useState<"live" | "upload" | "text">("live");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const m = MODES.find((x) => x.key === mode)!;
  const [, setWs] = useWorkspace();

  /** Guided demo: a scripted call played through the real analysis (backend demo.py). */
  const demo = async (scenario: string) => {
    setBusy(true); setErr(null);
    try {
      const r = await api.post<{ id: string }>("/api/demo/calls", { scenario });
      setWs("training");
      router.push(`/cases/${r.id}`);
    } catch (e: any) { setErr(e.message); setBusy(false); }
  };

  const go = async () => {
    setBusy(true); setErr(null);
    try {
      const c = await api.post<CaseDetail>("/api/cases", {
        mode, channel: m.channel, district: district.trim() || null, language,
        caller_label: caller.trim() || (mode === "suspect_recording" ? "Recording shared by complainant" : "Unnamed caller"),
      });
      router.push(`/cases/${c.id}?start=${start}`);
    } catch (e: any) { setErr(e.message); setBusy(false); }
  };

  return (
    <>
      <TopBar title="New intake" crumbs={[{ href: "/", label: "Case queue" }]} />
      <div className="mx-auto max-w-3xl space-y-5 p-6">
        <section className="panel border-accent-line">
          <header className="panel-h">
            <h3 className="panel-t">Guided demo call</h3>
            <span className="text-xs2 text-fg-3">Training workspace · never mixed with real cases</span>
          </header>
          <div className="flex flex-wrap items-center gap-3 px-4 py-3">
            <p className="min-w-[260px] flex-1 text-xs2 leading-[17px] text-fg-2">
              A scripted victim call plays out in about a minute. The words and the acted voice clips are scripted
              (one CREMA-D speaker the model never trained on); everything after that is the real analysis: facts,
              priority, distress, the automatic Tele-MANAS hand-off and linked cases.
            </p>
            <button onClick={() => demo("digital_arrest")} disabled={busy} className="btn-primary">Digital arrest (Hinglish)</button>
            <button onClick={() => demo("tamil_crisis")} disabled={busy} className="btn-secondary">Investment fraud (Tamil)</button>
          </div>
        </section>
        <section className="panel">
          <header className="panel-h"><h3 className="panel-t">1. Call type</h3><span className="text-xs2 text-fg-3">Confirmed by the operator</span></header>
          <div className="divide-y divide-line">
            {MODES.map((x) => (
              <label key={x.key} className={clsx("flex cursor-pointer gap-3 px-4 py-3", mode === x.key ? "bg-accent-soft/40" : "hover:bg-bg")}>
                <input type="radio" name="mode" className="mt-1 accent-[#F97A0F]" checked={mode === x.key}
                  onChange={() => { setMode(x.key); setStart(x.key === "suspect_recording" ? "upload" : "live"); }} />
                <div className="min-w-0">
                  <div className="font-medium text-fg">{x.title}</div>
                  <div className="text-fg-2">{x.desc}</div>
                  <div className="mt-1.5 grid gap-x-6 gap-y-0.5 text-xs2 sm:grid-cols-[64px_1fr]">
                    <span className="text-fg-3">Runs</span><span className="text-fg-2">{x.runs}</span>
                    <span className="text-fg-3">Disabled</span><span className="text-fg-2">{x.off}</span>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </section>

        <section className="panel">
          <header className="panel-h"><h3 className="panel-t">2. Details</h3></header>
          <div className="grid gap-4 p-4 sm:grid-cols-2">
            <label className="block">
              <span className="label">{mode === "suspect_recording" ? "Source" : "Caller reference"}</span>
              <input className="input mt-1" value={caller} onChange={(e) => setCaller(e.target.value)}
                placeholder={mode === "suspect_recording" ? "Recording shared by complainant" : "Name or reference number"} />
            </label>
            <label className="block">
              <span className="label">District</span>
              <input className="input mt-1" value={district} onChange={(e) => setDistrict(e.target.value)} placeholder="e.g. Pune" />
            </label>
            <label className="block">
              <span className="label">Caller&apos;s language</span>
              <select className="input mt-1" value={language} onChange={(e) => setLanguage(e.target.value)}>
                {LANGUAGES.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
              </select>
              <span className="mt-1 block text-xs2 text-fg-3">22 Indian languages, English and others. Leave on detect if unsure; the first seconds decide.</span>
            </label>
            <div className="sm:col-span-2">
              <span className="label block">Input</span>
              <div className="mt-1 inline-flex rounded border border-line2 p-0.5">
                {SOURCES.map((s) => (
                  <button key={s.key} onClick={() => setStart(s.key)}
                    className={clsx("h-7 rounded-[4px] px-3 text-[13px] font-medium", start === s.key ? "bg-brand text-white" : "text-fg-2 hover:text-fg")}>{s.label}</button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {err && <ErrorNote msg={err} />}
        <div className="flex items-center justify-between">
          <p className="max-w-md text-xs2 text-fg-3">Creating the case starts the intake timer and records an audit entry. Uploaded audio is hashed (SHA-256) and stored encrypted.</p>
          <button onClick={go} disabled={busy} className="btn-primary">{busy ? "Creating…" : "Create case"}</button>
        </div>
      </div>
    </>
  );
}

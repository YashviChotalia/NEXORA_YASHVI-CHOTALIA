"use client";

import { useEffect, useState } from "react";
import { TopBar } from "@/components/Shell";
import { ErrorNote, Panel } from "@/components/ui";
import { api } from "@/lib/api";

/** Every number here is read from a results file in the repository (eval/build_evidence.py), named on the page. */

interface Row {
  condition: string; accuracy_all: number; accuracy_answered: number; abstain: number; attention_recall: number;
  false_alarms: number; ece: number; raw_accuracy: number | null; recall: Record<string, number>;
}
interface Evidence {
  available: boolean; built: string; sources: Record<string, string>;
  ser?: { model: string; system: string; clips: number; rows: Row[]; fairness_gap: Record<string, number>; latency_ms: Record<string, number> };
  calls?: { calls: number; distressed_flagged: number; calm_flagged: number; calm_high: number; distressed_high: number; auc: number; raw_distressed_flagged: number; raw_auc: number };
  languages?: { clips: number; languages: number; anvesh: number; whisper: number; indic: number };
}

const pct = (v?: number | null, d = 0) => (v === null || v === undefined ? "—" : `${(100 * v).toFixed(d)}%`);

export default function EvidencePage() {
  const [e, setE] = useState<Evidence | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.get<Evidence>("/api/evidence").then(setE).catch((x) => setErr(x.message)); }, []);

  const phone = e?.ser?.rows.find((r) => r.condition.startsWith("Phone line"));
  const gap = e?.ser ? Math.max(...Object.values(e.ser.fairness_gap)) : null;

  return (
    <>
      <TopBar title="Evidence" right={<span className="text-xs2 text-fg-3">{e?.built ? `Built ${e.built} from the results files` : ""}</span>} />
      <div className="mx-auto max-w-5xl space-y-5 p-6">
        {err && <ErrorNote msg={err} />}
        {e && !e.available && <ErrorNote msg="No evaluation results bundled with this build (run eval/build_evidence.py)." />}
        {e?.available && (
          <>
            <p className="max-w-3xl text-[13px] leading-[19px] text-fg-2">
              Measured, not claimed. Speakers used for testing never appear in training; every choice (thresholds, the
              telephone head, which languages count as validated) was made on separate validation speakers first.
              These are acted recordings with a simulated phone line, not real helpline calls: treat them as the
              strongest evidence available before a field trial, not as field accuracy.
            </p>

            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              <Tile big={pct(e.calls?.distressed_flagged)} label="distressed callers flagged, whole calls on a phone line"
                sub={`raw model: ${pct(e.calls?.raw_distressed_flagged)} · ${e.calls?.calls} calls, held-out speakers`} />
              <Tile big={pct(e.calls?.calm_high)} label="calm callers pushed to “high” distress"
                sub={`calm calls flagged elevated (tie-break only): ${pct(e.calls?.calm_flagged)}`} />
              <Tile big={e.calls?.auc?.toFixed(2) ?? "—"} label="ranking quality (AUC), distressed vs calm calls"
                sub={`raw model: ${e.calls?.raw_auc?.toFixed(2)}`} />
              <Tile big={phone?.ece?.toFixed(2) ?? "—"} label="calibration error on phone audio (lower is better)"
                sub="a 70% confidence is right about 70% of the time" />
              <Tile big={pct(e.languages?.anvesh)} label={`right language picked, ${e.languages?.languages} Indian languages`}
                sub={`Whisper alone: ${pct(e.languages?.whisper)} · ${e.languages?.clips} clips`} />
              <Tile big={gap !== null ? `${(100 * gap).toFixed(1)} pts` : "—"} label="largest accuracy gap between speaker groups"
                sub="by sex, age band and ethnicity (CREMA-D demographics)" />
            </div>

            {e.ser && (
              <Panel title="Voice emotion on single clips: studio vs a telephone line" pad={false}
                action={<span className="text-xs2 text-fg-3">{e.ser.clips} held-out clips · {e.ser.model} + {e.ser.system}</span>}>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse text-[13px]">
                    <thead><tr>
                      {["Audio", "Accuracy (all clips)", "Raw model", "Attention recall", "False alarms (calm)", "Fear / sad recall", "Calibration error"].map((h) =>
                        <th key={h} className="th">{h}</th>)}
                    </tr></thead>
                    <tbody>
                      {e.ser.rows.map((r) => (
                        <tr key={r.condition}>
                          <td className="td font-medium">{r.condition}</td>
                          <td className="td tabular-nums">{pct(r.accuracy_all, 1)}</td>
                          <td className="td tabular-nums text-fg-3">{pct(r.raw_accuracy, 1)}</td>
                          <td className="td tabular-nums">{r.attention_recall.toFixed(2)}</td>
                          <td className="td tabular-nums">{pct(r.false_alarms, 1)}</td>
                          <td className="td tabular-nums">{r.recall.fearful?.toFixed(2)} / {r.recall.sad?.toFixed(2)}</td>
                          <td className="td tabular-nums">{r.ece.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="border-t border-line px-4 py-2.5 text-xs2 text-fg-3">
                  4 classes (neutral, anger, fear, sadness), chance = 25%. Phone audio is much harder: fear and sadness are
                  the weakest classes, which is why a single window never changes priority: only sustained distress over
                  several windows does, by at most one level. Median latency {e.ser.latency_ms.median} ms per window on a laptop CPU.
                </p>
              </Panel>
            )}

            <Panel title="How it was measured">
              <ul className="list-disc space-y-1 pl-5 text-[13px] text-fg-2">
                <li>Speaker-disjoint splits: test speakers (id divisible by 3) are never used to train or tune anything.</li>
                <li>The phone line is simulated: 8 kHz, 300–3400 Hz, G.711 µ-law, plus street-like (pink) noise at 20 / 10 / 5 dB.</li>
                <li>Whole calls are scored by the app&apos;s own distress code, the same 3-second windows as a live call.</li>
                <li>Languages: 22 Indian languages from AI4Bharat Rasa; voice emotion only affects priority in languages that passed a test fixed in advance.</li>
                <li>Limits: acted speech, one dataset per language, no real helpline calls yet. Emotion readings are indicative and never evidence of honesty, intent or guilt.</li>
              </ul>
              <div className="mt-3 border-t border-line pt-2 text-xs2 text-fg-3">
                Sources: {Object.values(e.sources).map((s) => <span key={s} className="mono mr-3 inline-block">{s}</span>)}
              </div>
            </Panel>
          </>
        )}
      </div>
    </>
  );
}

function Tile({ big, label, sub }: { big: string; label: string; sub?: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="text-[26px] font-semibold tabular-nums tracking-[-0.02em] text-fg">{big}</div>
      <div className="mt-0.5 text-[13px] text-fg-2">{label}</div>
      {sub && <div className="mt-1 text-xs2 text-fg-3">{sub}</div>}
    </div>
  );
}

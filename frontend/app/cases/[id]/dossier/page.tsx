"use client";

import clsx from "clsx";
import { ArrowLeft, Download } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Logo } from "@/components/Shell";
import { ErrorNote, PriorityLabel, StatusTag } from "@/components/ui";
import { api } from "@/lib/api";
import { inr, languageName, MODE_LABEL, SPEAKER_LABEL, ts } from "@/lib/format";
import type { CaseField, Distress, Handoff, Hit, LinkT, Priority, Recording, SegmentT, Triage, VoiceAuth } from "@/lib/types";

/** One printable page per case. The browser's "Save as PDF" turns it into the dossier file, so every Indian script
 *  in the transcript prints correctly without bundling fonts on the server. */

interface Dossier {
  title: string; classification: string; disclaimer: string; generated_by: string; generated_at: string;
  case: {
    id: string; mode: string; channel: string; caller_label: string; district?: string | null; created_at: string;
    call_started_at?: string; priority: Priority; priority_score: number; playbook: { label: string; confidence: number };
    language?: string | null; amount: number; identity_protected?: boolean; synthetic?: boolean; intake_seconds?: number | null;
  };
  fields: CaseField[]; triage: Triage; distress: Distress | null;
  safety: { crisis?: boolean; danger?: boolean; coercion?: boolean; women?: boolean; child?: boolean;
            crisis_hits?: Hit[]; danger_hits?: Hit[]; coercion_hits?: Hit[]; women_hits?: Hit[]; child_hits?: Hit[] } | null;
  handoffs: Handoff[]; segments: SegmentT[]; links: LinkT[]; recordings: Recording[]; voice_auth: VoiceAuth | null;
  tactics: { items?: { key: string; label: string; cues: string[] }[] } | null;
  trail: { id: number; ts: string; actor: string; action: string; hash: string }[];
  integrity: { fingerprint_sha256: string; audit_chain_ok: boolean; audit_events_checked?: number; audit_head?: string; pipeline: string };
}

const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "—");
const DISTRESS: Record<string, string> = {
  off: "Not assessed (suspect recording)", abstain: "No reading (audio or language not reliable enough)",
  low: "Low", elevated: "Elevated", high: "High",
};
const MISSING: Record<string, string> = {
  amount_lost: "amount lost", incident_time: "time of the last payment", upi_id: "recipient UPI ID", bank: "debited bank",
  phone_number: "scammer's number", transaction_id: "UTR / transaction ID", account_number: "account number",
};
const SAFETY: [key: "crisis" | "danger" | "coercion" | "women" | "child", label: string][] = [
  ["crisis", "Crisis words (suicide, self-harm, worthlessness)"], ["danger", "Danger to life right now"],
  ["coercion", "Possible active coercion"], ["women", "Women's safety"], ["child", "A child may be at risk"],
];

export default function DossierPage() {
  const { id } = useParams<{ id: string }>();
  const [d, setD] = useState<Dossier | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.get<Dossier>(`/api/cases/${id}/dossier`).then(setD).catch((e) => setErr(e.message)); }, [id]);
  useEffect(() => { if (d) document.title = `Anvesh dossier ${d.case.id}`; }, [d]);

  if (err) return <div className="p-6"><ErrorNote msg={err} /></div>;
  if (!d) return <div className="p-6 text-fg-3">Preparing the dossier…</div>;
  const c = d.case, tri = d.triage, dist = d.distress;
  const victimLike = c.mode !== "suspect_recording";
  const firstQuote = (f: CaseField) => f.evidence.map((e) => `${ts(e.t_start)} “${e.quote}”`).join("; ");

  return (
    <div className="dossier min-h-screen bg-bg py-6 print:bg-white print:py-0">
      <div className="no-print mx-auto mb-4 flex max-w-[860px] items-center justify-between px-4">
        <Link href={`/cases/${c.id}`} className="btn-quiet"><ArrowLeft size={14} /> Back to case</Link>
        <button onClick={() => window.print()} className="btn-primary"><Download size={14} /> Save as PDF</button>
      </div>

      <article className="mx-auto max-w-[860px] rounded-md border border-line bg-white px-10 py-8 text-[12.5px] leading-[18px] text-fg shadow-sm print:max-w-none print:rounded-none print:border-0 print:p-0 print:shadow-none">
        <header className="flex items-start justify-between border-b-2 border-fg pb-3">
          <div>
            <Logo height={20} />
            <h1 className="mt-2 text-[20px]">Case dossier <span className="mono text-[15px]">{c.id}</span></h1>
            <div className="text-xs2 text-fg-3">Generated {when(d.generated_at)} by {d.generated_by}</div>
          </div>
          <div className="text-right">
            <PriorityLabel p={c.priority} className="text-[15px]" />
            <div className="text-xs2 text-fg-3">priority score {Math.round(c.priority_score)}</div>
          </div>
        </header>
        <p className="mt-2 rounded-sm bg-fg px-2 py-1 text-center text-[11px] font-semibold uppercase tracking-wide text-white print:bg-fg">
          {d.classification}
        </p>
        {c.synthetic && <p className="mt-2 text-xs2 font-medium text-warn">Training scenario: synthetic case, not a real complainant.</p>}

        <Section n={1} title="Summary">
          <Grid rows={[
            ["Call type", MODE_LABEL[c.mode] || c.mode], ["Caller", c.caller_label],
            ["District", c.district || "—"], ["Language", languageName(c.language) || "—"],
            ["Incident type", c.playbook?.label || "—"], ["Reported loss", inr(c.amount)],
            ["Call started", when(c.call_started_at || c.created_at)],
            ["Intake time", c.intake_seconds ? `${Math.round(c.intake_seconds / 60)} min ${Math.round(c.intake_seconds % 60)} s` : "Intake not completed"],
          ]} />
          {c.identity_protected && <p className="mt-2 text-xs2 text-warn">Informant identity protected: share this dossier only with the intake officer and supervisors.</p>}
        </Section>

        <Section n={2} title="Priority and why">
          <p className="font-medium">{tri?.headline}</p>
          <ul className="mt-1 list-disc pl-5">{(tri?.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
          {tri?.emotion_adjustment && tri.emotion_adjustment.from !== tri.emotion_adjustment.to && (
            <p className="mt-1 text-fg-2">Voice distress adjustment: {tri.emotion_adjustment.from} → {tri.emotion_adjustment.to}. {tri.emotion_adjustment.reason}</p>
          )}
          {tri?.next_action && <p className="mt-2"><b>Next action:</b> {tri.next_action}</p>}
          {tri?.missing?.length ? <p className="mt-1 text-fg-2"><b>Still missing:</b> {tri.missing.map((m) => MISSING[m] || m.replace(/_/g, " ")).join(", ")}</p> : null}
        </Section>

        {victimLike && (
          <Section n={3} title="Human safety and helpline hand-offs">
            {SAFETY.filter(([k]) => d.safety?.[k]).length === 0
              ? <p className="text-fg-2">No safety signal in the caller&apos;s words.</p>
              : (
                <ul className="space-y-1">
                  {SAFETY.filter(([k]) => d.safety?.[k]).map(([k, label]) => (
                    <li key={k}><b>{label}:</b>{" "}
                      {onePerSegment(d.safety?.[`${k}_hits` as "crisis_hits"]).slice(0, 4).map((h, i) => (
                        <span key={i} className="mr-2">{ts(h.t_start)} “{h.quote}”</span>))}
                    </li>
                  ))}
                </ul>
              )}
            {d.handoffs.length > 0 && (
              <Table className="mt-3" head={["Helpline", "Number", "Outcome", "How", "By", "When"]} rows={d.handoffs.map((h) => [
                h.name, h.number, h.status + (h.detail ? ` · ${h.detail}` : ""),
                h.trigger === "auto" ? "Automatic (countdown)" : h.trigger === "officer" ? "Officer"
                  : h.status === "suggested" || h.status === "dismissed" ? "Suggested" : "Suggested, officer connected",
                h.actor || "—", when(h.executed_at || h.created_at)])} />
            )}
          </Section>
        )}

        <Section n={victimLike ? 4 : 3} title="Facts captured from the call">
          {d.fields.length === 0 ? <p className="text-fg-2">No facts captured yet.</p> : (
            <Table head={["Field", "Value", "Status", "Where it was said"]} rows={d.fields.map((f) => [
              f.label, <span key="v" className="mono">{f.display}</span>, <StatusTag key="s" s={f.status} />,
              <span key="e" className="text-fg-2">{firstQuote(f) || "Entered by officer"}</span>])} />
          )}
          {d.tactics?.items?.length ? (
            <p className="mt-2"><b>Scam tactics heard:</b> {d.tactics.items.map((t) => t.label).join(" · ")}</p>
          ) : null}
        </Section>

        <Section n={victimLike ? 5 : 4} title="Voice signals (indicative only)">
          <Grid rows={[
            ["Caller distress (voice and words)", dist ? (DISTRESS[dist.level] || dist.level) + (dist.score != null ? ` · ${dist.score.toFixed(2)}` : "") : "—"],
            ["Confidence", dist?.confidence || "—"],
            ["Synthetic-voice check", d.voice_auth?.verdict ? d.voice_auth.verdict.replace(/_/g, " ") : "Not run"],
            ["Recordings checked", String(d.voice_auth?.recordings_checked ?? d.recordings.length)],
          ]} />
          {dist && dist.timeline?.length > 1 && <Spark points={dist.timeline} />}
          {(dist?.reasons || []).length > 0 && <ul className="mt-1 list-disc pl-5 text-fg-2">{dist!.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
          {(dist?.abstain_reasons || []).length > 0 && <p className="mt-1 text-fg-3">No reading because: {dist!.abstain_reasons.join("; ")}</p>}
          <p className="mt-2 text-xs2 text-fg-3">These readings describe how the caller sounded, compared with their own voice earlier in the call. They are not evidence of honesty, intent or guilt.</p>
        </Section>

        {d.links.length > 0 && (
          <Section n={victimLike ? 6 : 5} title="Linked cases (leads to check)">
            <Table head={["Case", "Link", "Why", "Status"]} rows={d.links.map((l) => [
              <span key="i" className="mono">{l.other.id}</span>, l.kind, l.detail.reason || "—", l.status])} />
          </Section>
        )}

        <Section n={victimLike ? (d.links.length ? 7 : 6) : (d.links.length ? 6 : 5)} title="Transcript">
          <table className="w-full border-collapse">
            <tbody>
              {d.segments.map((s) => (
                <tr key={s.id} className="align-top">
                  <td className="mono w-[54px] border-b border-line py-1 pr-2 text-fg-3">{ts(s.start)}</td>
                  <td className="w-[78px] border-b border-line py-1 pr-2 font-medium">{SPEAKER_LABEL[s.speaker || "unknown"] || s.speaker}</td>
                  <td className="border-b border-line py-1">
                    {s.text_native && s.text_native !== s.text ? <><div>{s.text_native}</div><div className="text-fg-3">{s.text}</div></> : s.text}
                    {s.low_confidence && <span className="ml-1 text-xs2 text-warn">(low confidence: check the audio)</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Evidence integrity">
          <Grid rows={[
            ["Dossier fingerprint (SHA-256)", <span key="f" className="mono break-all">{d.integrity.fingerprint_sha256}</span>],
            ["Audit chain", d.integrity.audit_chain_ok ? `Intact · ${d.integrity.audit_events_checked ?? ""} events verified` : "BROKEN: the log was altered"],
            ["Audit head", <span key="h" className="mono break-all">{d.integrity.audit_head || "—"}</span>],
            ["Pipeline version", <span key="p" className="mono">{d.integrity.pipeline}</span>],
          ]} />
          {d.recordings.length > 0 && (
            <Table className="mt-3" head={["Recording", "Duration", "SHA-256"]} rows={d.recordings.map((r) => [
              r.filename, r.duration ? `${Math.round(r.duration)} s` : "—", <span key="h" className="mono break-all">{r.sha256}</span>])} />
          )}
          <div className="mt-3 label">Audit trail for this case</div>
          <Table head={["#", "Time", "Who", "Action", "Hash"]} rows={d.trail.map((e) => [
            String(e.id), when(e.ts), e.actor, e.action, <span key="h" className="mono">{e.hash.slice(0, 16)}…</span>])} />
          <p className="mt-2 text-xs2 text-fg-3">The fingerprint of this dossier is recorded in the audit log (action “dossier.generated”), so a printed copy can be matched to the log.</p>
        </Section>

        <footer className="mt-6 border-t border-line pt-3 text-xs2 text-fg-3">{d.disclaimer}</footer>
      </article>
    </div>
  );
}

/** A spelling-tolerant match quotes the whole line; drop it when an exact quote from the same line is inside it. */
function onePerSegment(hits?: Hit[] | null): Hit[] {
  const hs = hits || [];
  const keep = hs.filter((h, i) => !hs.some((o, j) => j !== i && o.segment_id === h.segment_id
    && o.quote.length < h.quote.length && h.quote.includes(o.quote)));
  return keep.filter((h, i) => keep.findIndex((o) => o.segment_id === h.segment_id && o.quote === h.quote) === i)
    .sort((a, b) => a.t_start - b.t_start);
}

function Section({ n, title, children, breakBefore }: { n?: number; title: string; children: React.ReactNode; breakBefore?: boolean }) {
  return (
    <section className={clsx("mt-5", breakBefore && "print:break-before-page")}>
      <h2 className="mb-2 break-after-avoid border-b border-line pb-1 text-[13.5px]">{n ? `${n}. ` : ""}{title}</h2>
      {children}
    </section>
  );
}

function Grid({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5">
      {rows.map(([k, v]) => <div key={k} className="flex gap-2"><dt className="w-[150px] shrink-0 text-fg-3">{k}</dt><dd className="min-w-0">{v}</dd></div>)}
    </dl>
  );
}

function Table({ head, rows, className }: { head: string[]; rows: React.ReactNode[][]; className?: string }) {
  return (
    <table className={clsx("w-full border-collapse", className)}>
      <thead><tr>{head.map((h) => <th key={h} className="border-b border-line py-1 pr-2 text-left text-xs2 font-medium text-fg-3">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="align-top break-inside-avoid">{r.map((x, j) => <td key={j} className="border-b border-line py-1 pr-2">{x}</td>)}</tr>)}</tbody>
    </table>
  );
}

/** Distress over the call, 0-1, as a small line: enough to see when it rose. */
function Spark({ points }: { points: { t: number; score: number }[] }) {
  const w = 760, h = 64, t1 = Math.max(...points.map((p) => p.t), 1);
  const xy = points.map((p) => `${(p.t / t1) * w},${h - Math.max(0, Math.min(1, p.score)) * h}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h + 14}`} className="mt-2 w-full" role="img" aria-label="Caller distress over the call">
      <line x1="0" x2={w} y1={h - 0.55 * h} y2={h - 0.55 * h} stroke="#D92D20" strokeDasharray="3 3" strokeWidth="1" />
      <line x1="0" x2={w} y1={h - 0.4 * h} y2={h - 0.4 * h} stroke="#DC6803" strokeDasharray="3 3" strokeWidth="1" />
      <polyline points={xy} fill="none" stroke="#101828" strokeWidth="1.6" />
      <text x="0" y={h + 12} fontSize="10" fill="#667085">00:00</text>
      <text x={w} y={h + 12} fontSize="10" fill="#667085" textAnchor="end">{ts(t1)}</text>
      <text x={w} y={h - 0.55 * h - 3} fontSize="10" fill="#D92D20" textAnchor="end">high</text>
      <text x={w} y={h - 0.4 * h - 3} fontSize="10" fill="#DC6803" textAnchor="end">elevated</text>
    </svg>
  );
}

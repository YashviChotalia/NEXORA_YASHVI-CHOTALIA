export type Mode = "victim" | "suspect_recording" | "informant";
export type Priority = "IMMEDIATE" | "HIGH" | "STANDARD";

export interface Evidence {
  segment_id: string; t_start: number; t_end: number; char_start: number; char_end: number; quote: string;
}
export interface CaseField {
  name: string; label: string; value: string; display: string; confidence: number;
  status: "predicted" | "reported" | "verified" | "corrected" | "missing";
  evidence: Evidence[]; note?: string | null; model_value?: string | null; updated_by?: string | null;
}
export interface SegmentT {
  id: string; start: number; end: number; text: string; text_native?: string | null; speaker?: string | null;
  language?: string | null; low_confidence: boolean; recording_id?: string | null;
}
export interface Triage {
  priority: Priority; score: number; headline: string; reasons: string[]; missing: string[];
  next_action: string; human_safety: string[]; evidence_status: string;
  financial_score?: number;
  /** set when voice distress changed the queue order (capped: one level, or a tie-break inside a level) */
  emotion_adjustment?: { from: Priority; to: Priority; points: number; reason: string } | null;
}
export interface Distress {
  enabled: boolean; level: "off" | "abstain" | "low" | "elevated" | "high"; score: number | null;
  confidence: string; timeline: { t: number; score: number; label?: string | null; p?: number | null }[]; reasons: string[]; abstain_reasons: string[]; label: string;
}
export interface Safety {
  coercion: boolean; crisis: boolean; coercion_hits: Hit[]; crisis_hits: Hit[];
  coercion_message?: string | null; crisis_message?: string | null; playbook_advice?: string;
}
export interface Hit { segment_id: string; phrase: string; quote: string; t_start: number; t_end: number }
export interface Prompt { key: string; question: string; kind: string; priority: number }
export interface CaseSummary {
  id: string; created_at: string; mode: Mode; channel: string; status: string; caller_label: string;
  district?: string | null; priority: Priority; priority_score: number; synthetic: boolean;
  playbook: { key: string | null; label: string; confidence: number }; amount: number;
  headline?: string; identity_protected?: boolean; voice_flag?: boolean; coercion: boolean; crisis: boolean; language?: string | null; intake_seconds?: number | null;
}
export interface LinkT {
  id: number; kind: "identifier" | "script" | "voice"; score: number; status: "candidate" | "confirmed" | "rejected";
  detail: { reason?: string; type?: string; value?: string; method?: string }; other: CaseSummary;
}
export interface Recording { id: string; kind: string; filename: string; sha256: string; duration?: number | null; size_bytes: number; has_voiceprint: boolean; offset?: number; voice_auth?: VoiceAuth | null }
export interface VoiceAuth {
  verdict: string; score: number | null; model_score: number | null; cue_score: number | null; model?: string | null;
  cues?: Record<string, number | null>; reasons: string[]; abstain: string[]; advice: string | null; recordings_checked?: number; caller_class?: string;
}
/** A warm transfer of the call to a helpline (Tele-MANAS 14416, 112, 181, 1098, 1930). */
export interface Handoff {
  id: string; helpline: string; name: string; number: string; reason: string | null;
  status: "suggested" | "pending" | "connecting" | "connected" | "cancelled" | "dismissed" | "failed";
  trigger: "auto" | "suggest" | "officer"; mode: string; created_at: string | null; execute_at: string | null;
  executed_at: string | null; actor: string | null; provider_ref: string | null; detail: string | null;
}
export interface CaseDetail extends CaseSummary {
  call_started_at: string; segments: SegmentT[]; fields: CaseField[]; triage: Triage; distress: Distress;
  safety: Safety; tactics: { items?: { key: string; label: string; cues: string[] }[]; note?: string };
  copilot: Prompt[]; recordings: Recording[]; links: LinkT[]; processing?: string | null;
  handoffs?: Handoff[];
  /** a live call is streaming into this case right now (possibly from another device) */
  live?: boolean;
  local_context?: LocalContext | null; voice_auth?: VoiceAuth | null; official_scam_type?: ScamTypeContext | null; scenario?: Scenario | null;
}
export interface SourceRef { key?: string; title: string; publisher?: string; report: string; page: string; reference?: string; note?: string; sha256?: string; retrieved?: string }
export interface StateStat {
  name: string; cases_2021: number; cases_2022: number; cases_2023: number; change_2022_2023_pct: number | null;
  rate_2023: number | null; chargesheet_rate_2023: number | null; fraud_2023: number | null; fraud_share_2023_pct: number | null;
  extortion_2023?: number | null; sexual_exploitation_2023?: number | null;
}
export interface LocalContext {
  district: string; state: StateStat; india: StateStat; sources: SourceRef[];
  city?: { name: string; cases_2023: number; fraud_2023: number; fraud_share_2023_pct: number | null };
}
export interface ScamTypeContext { categories: string[]; reports_2025: number; reports_2024: number; share_2025_pct: number | null; source: SourceRef }
export interface Scenario {
  kind: string; note: string; category: string; category_source: SourceRef;
  locality?: { district: string; state: string; state_cyber_crimes_2023: number; source: SourceRef } | null;
}
export interface OfficialSummary {
  headline: {
    cyber_crimes_2023: number; cyber_crimes_2022: number; growth_2022_2023_pct: number; growth_2019_2023_x: number;
    fraud_share_2023_pct: number; chargesheet_rate_2023: number; rate_2023: number;
    ncrp_complaints_latest: number; ncrp_year_latest: number; ncrp_amount_crore_latest: number;
    ncrp_amount_crore_prev: number; ncrp_year_prev: number; ncrp_complaints_growth_x: number; ncrp_amount_total_crore: number;
  };
  scam_types: { category: string; playbook: string | null; y2024: number; y2025: number; y2026_partial: number; share_2025_pct: number }[];
  scam_types_total: Record<string, number>;
  trend: { year: number; cases: number; fraud_ipc: number; cheating_ipc: number; computer_related_it_act: number }[];
  ncrp: { year: number; complaints: number; amount_crore: number }[];
  motives_2023: { motive: string; cases: number; share_pct: number }[];
  states: StateStat[];
  cities: { name: string; cases_2023: number; fraud_2023: number; fraud_share_2023_pct: number | null }[];
  sources: SourceRef[];
}
export interface Ring {
  id: string; cases: string[]; size: number; districts: string[]; total_amount: number; shared_identifiers: string[];
  link_kinds: string[]; confirmed: number; candidates: number; playbooks: string[]; synthetic: boolean;
}
export interface Stats {
  open_cases: number; immediate: number; safety_alerts: number; rings: number; linked_cases: number;
  reported_loss: number; median_intake_seconds: number | null; last_24h: number;
  workspace?: "live" | "training"; training_available?: number;
}
export interface AuditEventT { id: number; ts: string; actor: string; action: string; case_id: string | null; payload: string; hash: string; prev_hash: string; model_version?: string | null }
export interface Health { ok: boolean; version: string; models: { name: string; model: string; loaded: boolean; state?: string; error?: string | null; license?: string | null }[]; validated_languages: string[] }

export function inr(v: number | null | undefined): string {
  if (!v) return "—";
  return "₹" + Math.round(v).toLocaleString("en-IN");
}

export function inrShort(v: number): string {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(v >= 1e8 ? 0 : 1)} Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(v >= 1e6 ? 0 : 1)} L`;
  if (v >= 1e3) return `₹${(v / 1e3).toFixed(0)}K`;
  return `₹${Math.round(v)}`;
}

export function ts(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function ago(iso: string): string {
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return `${Math.floor(d / 86400)} d ago`;
}

export const MODE_LABEL: Record<string, string> = {
  victim: "Victim report",
  suspect_recording: "Suspected scam recording",
  informant: "Informant tip",
};

export const SPEAKER_LABEL: Record<string, string> = {
  caller: "Caller", officer: "Officer", suspect: "Suspect", informant: "Informant", unknown: "Unknown",
};

/** Languages Anvesh can transcribe: Whisper (English, Hindi, foreign) and IndicConformer (22 Indian languages). */
export const LANGUAGES: [string, string][] = [
  ["auto", "Detect automatically"], ["hi", "Hindi / Hinglish"], ["en", "English"],
  ["as", "Assamese"], ["bn", "Bengali"], ["brx", "Bodo"], ["doi", "Dogri"], ["gu", "Gujarati"], ["kn", "Kannada"],
  ["ks", "Kashmiri"], ["kok", "Konkani"], ["mai", "Maithili"], ["ml", "Malayalam"], ["mni", "Manipuri"],
  ["mr", "Marathi"], ["ne", "Nepali"], ["or", "Odia"], ["pa", "Punjabi"], ["sa", "Sanskrit"], ["sat", "Santali"],
  ["sd", "Sindhi"], ["ta", "Tamil"], ["te", "Telugu"], ["ur", "Urdu"],
  ["ar", "Arabic"], ["zh", "Chinese"], ["fr", "French"], ["de", "German"], ["ru", "Russian"], ["es", "Spanish"],
];
export const languageName = (code?: string | null) =>
  (code && LANGUAGES.find(([c]) => c === code)?.[1]) || code || "";


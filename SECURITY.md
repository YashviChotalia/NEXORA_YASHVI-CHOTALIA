# Anvesh security model

Anvesh holds some of the most sensitive data a police unit handles: what victims say on a call, their phone
numbers, bank and UPI details, recordings of their voice, and the identity of informants. This document says
what we protect, from whom, how, and how it is tested.

## What we protect

| Asset | Why it matters |
|---|---|
| Call transcripts, caller names, extracted details (UPI IDs, accounts, numbers) | Personal data of victims (DPDP Act 2023); can be used to re-victimise them |
| Call recordings and suspect voiceprints | Evidence, and biometric data |
| Informant identity | Disclosure can put a person in danger |
| Case links between victims | Reveal who else was targeted |
| The audit trail | Proves who saw or changed evidence |

## Threats considered

1. Anyone on the network reading or changing cases without signing in.
2. An insider seeing more than their job needs (an operator browsing informants, an IT admin reading cases).
3. Theft of the database file, a backup, or the laptop's data folder.
4. Tampering with evidence or the audit log after the fact, including by someone with database access.
5. Browser attacks: CSRF, cross-site WebSocket hijacking, XSS, clickjacking, caching of personal data.
6. Password guessing and session theft.
7. Abuse of inputs: oversized uploads, path traversal, injection strings.
8. Known-vulnerable dependencies.

## Controls

**Identity and access**
- Named accounts only. First-run setup creates one administrator and is possible once, from the server computer itself.
- Passwords: scrypt (N=2^15, r=8, p=1), minimum 12 characters, three character classes, common and username-based passwords rejected. New and reset accounts must set their own password before doing anything else.
- Sessions: 256-bit random token in an `HttpOnly`, `SameSite=Strict` cookie (`Secure` over HTTPS). The server stores only an HMAC of it, so a stolen database cannot be replayed as a login. 30-minute idle timeout, 12-hour maximum, server-side revocation on logout, password change, role change or disable.
- Brute force: account lockout for 15 minutes after 5 failures, 10 sign-in attempts per minute per client, identical error for wrong password and unknown user (no username oracle), constant work for unknown users.
- Roles, least privilege and separation of duties:

| Permission | Operator | Supervisor | Administrator |
|---|:-:|:-:|:-:|
| Take calls, work cases | ✓ | ✓ | |
| Review case links, erase a case | | ✓ | |
| See a protected informant's identity | own intakes | ✓ | |
| Read the audit log | | ✓ | ✓ |
| Manage accounts | | | ✓ |

  An administrator manages people but cannot read case content unless also given a case role. The actor recorded for every action comes from the signed-in session; client-supplied names are ignored.
- Every route except health, sign-in, setup and auth state requires a session; a test enumerates all routes to keep it that way.

**Confidentiality at rest**
- Field-level encryption, AES-256-GCM, for caller names, narratives, transcript text (Roman and original script), word timings, extracted values and their evidence quotes, identifiers, link reasons, triage/safety/distress details, voice-emotion windows, recording filenames and speaker embeddings. The column name is bound as associated data, so ciphertext cannot be moved between columns.
- Identifiers used to link cases (UPI IDs, phone numbers, accounts) are matched through an HMAC blind index; the plaintext is never stored or indexed.
- Evidence files: Fernet (AES-128-CBC + HMAC-SHA256) with a derived key, authenticated on every read, with the SHA-256 of the original audio checked against the case record. Files are named by a keyed hash, so the name does not reveal the content hash. Paths are confined to the evidence folder.
- Keys: one 256-bit master key, supplied as `ANVESH_MASTER_KEY` (from a KMS or secret manager in production) or generated in `~/.anvesh/keys/master.key`, outside the data folder, so a copied database or backup alone is unreadable. Separate keys for fields, blind index, audit, evidence and sessions are derived with HKDF-SHA256. Data and evidence from the earlier version are encrypted in place on first start, and the old key file that sat beside the data is retired.
- Back up the master key separately from the data. Without it the encrypted data cannot be recovered.

**Integrity and accountability**
- The audit log is a hash chain signed with HMAC-SHA256, so someone with write access to the database alone cannot edit a row and recompute a valid chain. Events from before keying are verified with SHA-256 and reported as such.
- The audit log never contains personal data: sensitive values are replaced with a keyed fingerprint, so it proves what changed and who changed it without becoming a second copy of the case file.
- Access logging: case views (at most once per officer per case per 10 minutes, noting when an informant identity was revealed), evidence playback, report generation, sign-ins, failed sign-ins, lockouts, user and role changes, integrity failures and erasures.
- Complaint drafts carry a confidentiality marking and the name of the officer who generated them.

**Browser and transport**
- Strict CORS: exact frontend origins only, with credentials, limited methods and headers; `*` is refused.
- CSRF: `SameSite=Strict` plus a per-session token required on every state-changing request; the token is kept in memory, never in storage.
- WebSocket: the session cookie and an allowed `Origin` are required at the handshake; frames are capped at 256 KB and sessions at 120 minutes.
- API headers: `Cache-Control: no-store`, `Content-Security-Policy: default-src 'none'`, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, COOP/CORP, and HSTS over HTTPS. The frontend sends a CSP, frame-ancestors none and a microphone-only permissions policy.
- No secret is ever shipped to the browser (the old `NEXT_PUBLIC_API_TOKEN` design is removed).

**Input handling**
- Strict validation of IDs (path traversal and injection strings are rejected before the database), field lengths, segment counts, roles and usernames. All SQL goes through parameterised queries.
- Upload limits: 100 MB for audio, 512 KB for JSON, audio content types only, files that fail to decode are refused, filenames sanitised.

**Data minimisation and retention**
- Informant identity is hidden from staff without need-to-know.
- Supervisors can erase a case: transcript, fields, identifiers, links and evidence (overwritten, then deleted). The audit trail is kept, with a redacted erasure record.
- `ANVESH_RETENTION_DAYS` automatically erases closed cases older than N days.
- Training scenarios are kept apart from live cases and never link to them.

### Helpline hand-off (telephony)

- An automatic transfer only starts on a **live** victim or informant call, only for Tele-MANAS 14416 and 112, and
  only after a countdown (default 10 s) that any officer on the case can cancel. It is a three-way conference, never
  a blind transfer.
- In `webhook` mode the request to the phone system is signed (`X-Anvesh-Signature`, HMAC-SHA256 of the body with
  `ANVESH_TELEPHONY_SECRET`) and carries only the call reference, the helpline number and the case id; no
  transcript or audio leaves Anvesh. The reason text, which quotes the caller, is stored encrypted.
- Start, suggestion, connection, failure and cancellation are each written to the hash-chained audit log with
  the acting officer (or `system:auto-handoff`).

## Testing

- `backend/tests/test_security.py` contains 38 security tests, run in CI on every push. They cover anonymous access to every route, impersonation, cookie flags and hashed tokens, logout replay, idle expiry, password hashing and policy, lockout and no username oracle, rate limits, one-time setup, forced password change, session kill on disable, the role matrix, training-case abuse, informant masking, CSRF, CORS, plaintext absent from the raw database file, ciphertext bound to its column, blind-index linking, key location, evidence encryption, tamper detection and path confinement, audit privacy, audit tamper and re-forge detection, access logging, erasure, security headers, malformed IDs, injection strings, size and type limits, filename sanitising, WebSocket authentication and origin checks, no secrets in the frontend, and the plaintext-to-encrypted upgrade.
- Static and dependency checks in CI: `bandit` (no medium or high findings), `pip-audit` (no known vulnerabilities in the pinned backend dependencies), and `npm audit --omit=dev --audit-level=high` (none in the production frontend).

## Known limits and next steps

- The rate limiter keeps its state in memory: fine for one server process; use Redis when running several workers.
- Run behind HTTPS in any deployment beyond one computer, so cookies get `Secure` and HSTS applies.
- Master key in a hardware-backed store (Windows DPAPI, cloud KMS or HSM) and periodic key rotation.
- Database-level encryption (SQLCipher or encrypted PostgreSQL volumes) as a second layer.
- Multi-factor sign-in for supervisors and administrators.
- `postcss-selector-parser` (moderate) remains in Tailwind's build-time tooling; it processes only our own CSS and is not shipped to users.

## Reporting a vulnerability

Please email the maintainers privately rather than opening a public issue.

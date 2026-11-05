# Licence inventory

Verify every entry against the exact release you deploy. "Research" means fine for the hackathon demo;
it does not grant commercial rights.

| Asset | Used for | Licence (as understood) | Commercial use | Status |
|---|---|---|---|---|
| faster-whisper | ASR runtime | MIT | Yes | In use |
| Whisper weights (OpenAI) | ASR model | MIT | Yes | In use |
| emotion2vec+ (`iic/emotion2vec_plus_*`) | Voice emotion | "model-license" on the model card | **Unverified** | In use (research); verify before sale |
| FunASR | emotion2vec loader | MIT (code) | Yes (code) | In use |
| SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`) | Speaker embeddings | Apache-2.0 | Yes | Optional |
| sentence-transformers multilingual MiniLM | Script similarity | Apache-2.0 | Yes | Optional |
| scikit-learn, FastAPI, SQLAlchemy, Next.js, React | Application | BSD / MIT | Yes | In use |
| RAVDESS | Possible fine-tuning | CC BY-NC-SA 4.0 | **No** | Not used in product |
| IEMOCAP | Possible evaluation | Academic licence | **No** | Not used |
| CREMA-D | Trains the telephone-adapted emotion head; evaluation | Open Database Licence (ODbL) v1.0 | Likely, check terms | In use (head weights in `anvesh/data/models/`; no audio in the repo) |
| Anvesh telephone-adapted head (`ser_head_telephone.json`) | Voice emotion on phone audio | Anvesh code licence; built from CREMA-D (ODbL): attribution to CREMA-D, Cao et al. 2014 | Follow CREMA-D and emotion2vec+ terms | In use |
| ICFD-31k | Scam-text development | "other" | Unknown | Not used yet |

No customer audio is used for training by default.

## Official statistics (backend/anvesh/data/official)

Published by Government of India bodies on the Open Government Data Platform (data.gov.in) under the
Government Open Data License – India (GODL), which permits use and redistribution with attribution.
Attribution for each file (publisher, report, page URL, SHA-256) is in `sources.json` and shown in the UI.

| File | Publisher | Report |
|---|---|---|
| NCRB_CII_2023_Table_9A.1.csv | NCRB, Ministry of Home Affairs | Crime in India 2023, Table 9A.1 |
| NCRB_CII_2023_Table_9A.3.csv | NCRB, Ministry of Home Affairs | Crime in India 2023, Table 9A.3 |
| NCRB_CII_2023_Table_9B.3.csv | NCRB, Ministry of Home Affairs | Crime in India 2023, Table 9B.3 |
| RS_Session_269_AU_1185_… | Rajya Sabha (NCRB data) | Session 269, USQ 1185 |
| RS_Session_270_AU_1341_… | Rajya Sabha (I4C NCRP / CFCFRMS data) | Session 270, USQ 1341, 11 Feb 2026 |
| RS_Session_270_AU_647_… | Rajya Sabha (DoT Chakshu data) | Session 270, USQ 647, 5 Feb 2026 |

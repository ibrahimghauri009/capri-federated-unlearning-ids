# Dataset log

Raw files are kept in `data/raw/` (not committed). Hashes are SHA-256.

## Edge-IIoTset (Ferrag et al., 2022)

- Source: dataset authors' Kaggle upload `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`
- File used: `DNN-EdgeIIoT-dataset.csv`, 2,219,201 records, 63 columns
- SHA-256: `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7` (checked by `data.py` on every load)

## CICIoT2023 (Neto et al., 2023)

- Source: Canadian Institute for Cybersecurity, University of New Brunswick, official download (request form at cicresearch.ca), downloaded 2026-11-13
- `MERGED_CSV.zip`: 1,702,032,816 bytes, SHA-256 `7900eafc6edddd2cfae0b7ac32c62213f98babcd005a23d3fb8360c262c412e2`
- `README.pdf`: 3,819,227 bytes, SHA-256 `0f48daba395be03985f612ce706d33f1a25e4008cb94c3ad7b6f6332fbdbee92`
- Extracted: 63 files `Merged01.csv`–`Merged63.csv`, 9,300,139,459 bytes, 45,019,243 records, 40 columns (39 features + `Label`), 34 labels (benign + 33 attacks)
- Per-file sizes, row counts and hashes: `results/ciciot2023_inventory.json`

### Records per label (before any cleaning)

| Label | Records |
|---|---|
| DDOS-ICMP_FLOOD | 6,893,259 |
| DDOS-UDP_FLOOD | 5,181,027 |
| DDOS-TCP_FLOOD | 4,306,086 |
| DDOS-PSHACK_FLOOD | 3,920,372 |
| DDOS-SYN_FLOOD | 3,886,130 |
| DDOS-RSTFINFLOOD | 3,872,808 |
| DDOS-SYNONYMOUSIP_FLOOD | 3,445,659 |
| DOS-UDP_FLOOD | 3,177,323 |
| DOS-TCP_FLOOD | 2,558,256 |
| DOS-SYN_FLOOD | 1,942,176 |
| BENIGN | 1,051,373 |
| MIRAI-GREETH_FLOOD | 949,381 |
| MIRAI-UDPPLAIN | 852,695 |
| MIRAI-GREIP_FLOOD | 719,655 |
| DDOS-ICMP_FRAGMENTATION | 433,157 |
| VULNERABILITYSCAN | 357,583 |
| MITM-ARPSPOOFING | 294,469 |
| DDOS-UDP_FRAGMENTATION | 274,909 |
| DDOS-ACK_FRAGMENTATION | 272,793 |
| DNS_SPOOFING | 171,468 |
| RECON-HOSTDISCOVERY | 128,677 |
| RECON-OSSCAN | 93,970 |
| RECON-PORTSCAN | 78,730 |
| DOS-HTTP_FLOOD | 68,799 |
| DDOS-HTTP_FLOOD | 27,597 |
| DDOS-SLOWLORIS | 22,400 |
| DICTIONARYBRUTEFORCE | 12,522 |
| BROWSERHIJACKING | 5,630 |
| COMMANDINJECTION | 5,168 |
| SQLINJECTION | 5,022 |
| XSS | 3,705 |
| BACKDOOR_MALWARE | 3,078 |
| RECON-PINGSWEEP | 2,161 |
| UPLOADING_ATTACK | 1,196 |

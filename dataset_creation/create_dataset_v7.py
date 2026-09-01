"""
Version 7 Fraud Detection Dataset Curation Script
==========================================
"""

import hashlib
import json
import random
import re

# ── System prompt ─────────────────────────────────────────────────────────────

CANONICAL_SYSTEM = "You are a fraud detection expert system analyzing transaction risk."

# ── Threshold constants ───────────────────────────────────────────────────────

UNVERIFIED_DEVICE_THRESHOLD = 3000.0  # unverified device above this → HIGH
TELEMETRY_GAP_THRESHOLD = 3500.0  # unknown device/IP above this → HIGH
VPN_MEDIUM_AMOUNT_THRESHOLD = (
    1500.0  # VPN/Proxy above this → MEDIUM (or HIGH if Crypto)
)
WIRE_MEDIUM_AMOUNT_THRESHOLD = (
    2000.0  # Wire Transfer above this (clean profile) → MEDIUM
)
CRYPTO_MEDIUM_AMOUNT_THRESHOLD = 1000.0  # Crypto above this (clean profile) → MEDIUM

# ── IP risk classification ────────────────────────────────────────────────────

HIGH_RISK_IPS = frozenset(["Blacklisted IP", "Tor Exit Node"])
MEDIUM_RISK_IPS = frozenset(["VPN Datacenter", "Unknown Proxy"])

# ── Target class ratios  [FIX: EVAL-1] ───────────────────────────────────────
# Eval showed MEDIUM severely undertrained (F1=0.567). Raising share from
# ~22% → 35% while trimming HIGH from ~48% → 40% (HIGH F1 was already 0.935).
TARGET_HIGH_RATIO = 0.40
TARGET_MEDIUM_RATIO = 0.35
TARGET_LOW_RATIO = 0.25
# Boundary records: LOW records near each MEDIUM threshold (~5% of total).
# Hard-negative contrast that teaches the model exactly where each line sits.
BOUNDARY_RATIO = 0.05

# ── MEDIUM trigger rotation  [FIX: EVAL-1] ───────────────────────────────────
# generate_medium_record() cycles through these so all five MEDIUM risk paths
# receive equal representation in the generated portion of the dataset.
MEDIUM_TRIGGER_CYCLE = [
    "vpn_payment",  # VPN/Proxy + amount > $1,500, non-Crypto → vpn_medium
    "wire_clean_payment",  # Wire + clean IP + amount > $2,000        → wire_medium
    "crypto_clean_payment",  # Crypto + clean IP + amount > $1,000      → crypto_medium
    "chat_vpn_benign",  # benign payload + VPN/Proxy chat          → chat_ip_medium
    "unknown_proxy_payment",  # Unknown Proxy + amount > $1,500          → vpn_medium
]

# ── Vocabularies ──────────────────────────────────────────────────────────────

IP_TYPES = [
    "Blacklisted IP",
    "Tor Exit Node",
    "VPN Datacenter",
    "Unknown Proxy",
    "Clean IP",
    "Residential ISP",
]
DEVICE_TYPES = [
    "Mobile (verified)",
    "Tablet (unverified)",
    "Desktop (matched)",
    "Laptop (new)",
]
PAYMENT_METHODS = ["Credit Card", "Wire Transfer", "Crypto", "PayPal"]

# Verified-only devices — used in targeted generators to avoid accidentally
# firing the unverified-device HIGH path when we want MEDIUM/LOW output.
_VERIFIED_DEVICES = [d for d in DEVICE_TYPES if "unverified" not in d.lower()]

# ── Chat payload catalogues  [FIX: EVAL-8] ───────────────────────────────────
# Expanded from 4 → 8 entries each. Narrow vocabulary was causing the model to
# miss phishing patterns it hadn't seen at inference time (8 records scored 0.00
# on malicious payloads in the V6 eval).

MALICIOUS_PAYLOADS = [
    (
        "Urgently click this link to claim a 500 USDT airdrop:"
        " https://crypto-airdrop-claim.xyz"
    ),
    (
        "Security alert: Your account has been suspended. Verify identity now"
        " at https://bit.ly/secure-login-302"
    ),
    (
        "Congratulations! You won $1,000 in our giveaway. Claim here:"
        " https://reward-claim-portal.com"
    ),
    "Urgent! Transfer 0.1 BTC to verify your account address immediately.",
    (
        "Your crypto wallet has been compromised. Restore access now:"
        " https://wallet-restore-secure.net"
    ),
    (
        "URGENT: Unusual sign-in detected on your account. Verify immediately:"
        " https://verify-account-now.xyz"
    ),
    (
        "You have a pending refund of $847.50. Claim before it expires:"
        " https://refund-portal-us.com"
    ),
    (
        "Your package requires customs clearance. Pay the $3.99 release fee:"
        " https://pkg-customs-clearance.net"
    ),
]

BENIGN_PAYLOADS = [
    "Hey, are we still meeting for lunch at 12:30 PM today?",
    "Thanks for grabbing tickets! I just sent you $25 on Venmo.",
    "Can you send over the updated project document when you get a chance?",
    "Got it, see you at the office tomorrow morning!",
    "The meeting has been moved to 3 PM — let me know if that works.",
    "I'll pick up the kids from school today, no need to rush.",
    "Just submitted the quarterly report. All good on my end.",
    "Running late — grab us a table, I'll be there in 15.",
]

# ── Phantom attribute patterns (must not appear in response unless in prompt) ─

PHANTOM_PATTERNS = [
    r"remote\s+shipping",
    r"shipping\s+matches\s+billing",
    r"address\s+mismatch",
    r"velocity(\s+of\s+transactions)?",
    r"biometric(\s+matched)?(\s+device)?",
    r"account\s+takeover",
    r"credential\s+harvesting",
    r"compromised\s+device",
    r"credential\s+leak",
]

# ── Score bands ───────────────────────────────────────────────────────────────
#
#   LOW     : [0.01, 0.25]   no significant risk signals
#   MEDIUM  : [0.40, 0.65]   soft signals — warrants manual review    ← NEW
#   HIGH    : [0.80, 0.95]   single strong trigger
#   CERTAIN : [0.96, 0.99]   compound multi-trigger scenarios          ← NEW
#
#   Dead zones between tiers prevent any in-band ambiguity:
#     0.26 – 0.39  (gap below MEDIUM)
#     0.66 – 0.79  (gap above MEDIUM)

SCORE_BAND: dict[str, tuple[float, float]] = {
    "low": (0.01, 0.24),
    "medium": (0.40, 0.25),
    "high": (0.80, 0.15),
    "certain": (0.96, 0.03),
}

VERDICT_FOR_BAND: dict[str, str] = {
    "low": "LOW RISK (APPROVED)",
    "medium": "MEDIUM RISK (REVIEW)",
    "high": "HIGH RISK (FLAGGED)",
    "certain": "HIGH RISK (FLAGGED)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic profile hasher  [FIX: CRITICAL-1]
# ─────────────────────────────────────────────────────────────────────────────


def profile_hash(device: str, ip_status: str, risk_path: str) -> float:
    """
    Returns a float in [0.0, 1.0) derived deterministically from semantic
    profile features. Identical (device, ip_status, risk_path) triples always
    produce the same value, even after dataset shuffle, eliminating the score
    variance that the idx-based hasher introduced in V5.
    """
    key = f"{device}|{ip_status}|{risk_path}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 / 1000.0


def score_for_band(band: str, device: str, ip_status: str, risk_path: str) -> float:
    """
    Computes a deterministic risk score within the named band.
    Output is clamped to [0.01, 0.99] so a score of 0.00 is impossible
    regardless of hash edge cases.  [FIX: EVAL-3]
    """
    base, spread = SCORE_BAND[band]
    raw = base + profile_hash(device, ip_status, risk_path) * spread
    return round(max(0.01, min(0.99, raw)), 2)


# ── Valid threshold values (used by validate_format) ─────────────────────────
# Any dollar amount appearing next to a threshold keyword in a response must
# match one of these exactly (±0.01).  Prevents hallucinated thresholds like
# $300.00 that appeared in 4 records of the V6 eval.  [FIX: EVAL-6]
_VALID_THRESHOLD_AMOUNTS = frozenset(
    [
        UNVERIFIED_DEVICE_THRESHOLD,
        TELEMETRY_GAP_THRESHOLD,
        VPN_MEDIUM_AMOUNT_THRESHOLD,
        WIRE_MEDIUM_AMOUNT_THRESHOLD,
        CRYPTO_MEDIUM_AMOUNT_THRESHOLD,
    ]
)

# Canonical verdict strings — the only three the Final Assessment field may use.
_CANONICAL_VERDICTS = frozenset(
    [
        "HIGH RISK (FLAGGED)",
        "MEDIUM RISK (REVIEW)",
        "LOW RISK (APPROVED)",
    ]
)


def validate_format(record: dict) -> bool:
    """
    Post-generation format guard applied to every record — both ingested V4
    records and freshly synthesised ones — before it enters the dataset.

    Rejects on any of the following conditions:

      [EVAL-4]  Chat record has Step 4 labelled anything other than
                "Synthesis & Reasoning".
      [EVAL-5]  Any line in the assistant turn starts with a markdown list
                marker (-, •, *), indicating bullet-point payload analysis.
      [EVAL-6]  A dollar amount appearing next to a threshold keyword does not
                match one of the five declared threshold constants.
      [EVAL-7]  The Final Assessment field is absent or contains a non-canonical
                verdict string (e.g. "FLAGGED" instead of "HIGH RISK (FLAGGED)").
      [EVAL-3]  Risk score is outside (0.01, 0.99) — catches 0.00 artifacts.
    """
    msgs = record.get("messages", [])
    if len(msgs) != 3:
        return False

    user = msgs[1].get("content", "")
    asst = msgs[2].get("content", "")
    is_chat = "In-App Chat" in user

    # ── [EVAL-4] Step 4 label ─────────────────────────────────────────────────
    if is_chat and "Step 4:" in asst:
        if not re.search(r"Step 4:\s*Synthesis & Reasoning", asst):
            return False

    # ── [EVAL-5] Bullet points ────────────────────────────────────────────────
    if re.search(r"^\s*[-•*]\s+\S", asst, re.MULTILINE):
        return False

    # ── [EVAL-6] Hallucinated threshold values ────────────────────────────────
    for m in re.finditer(
        r"\$([\d,]+(?:\.\d+)?)\s+(?:[\w-]+\s+)?threshold"
        r"|threshold\s+of\s+\$([\d,]+(?:\.\d+)?)",
        asst,
        re.IGNORECASE,
    ):
        raw_val = (m.group(1) or m.group(2) or "").replace(",", "")
        if raw_val:
            try:
                val = float(raw_val)
                if not any(abs(val - t) < 0.01 for t in _VALID_THRESHOLD_AMOUNTS):
                    return False
            except ValueError:
                return False

    # ── [EVAL-7] Canonical verdict string ─────────────────────────────────────
    verdict_match = re.search(r"Final Assessment:\s*(.+?)\.", asst)
    if not verdict_match:
        return False
    if verdict_match.group(1).strip() not in _CANONICAL_VERDICTS:
        return False

    # ── [EVAL-3] Score range ──────────────────────────────────────────────────
    score_match = re.search(r"risk score of ([\d.]+)", asst)
    if score_match:
        score = float(score_match.group(1))
        if not (0.01 <= score <= 0.99):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Risk evaluation — payment  [FIX: CRITICAL-2, MODERATE-2/3, MINOR-1/2]
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_payment_risk(
    amount: float,
    method: str,
    device: str,
    ip_status: str,
) -> tuple[str, str, str, str]:
    """
    Evaluates payment-transaction risk.

    Returns (band, risk_path, verdict, reason).

    Priority order (highest wins):
        1. Compound triggers          → CERTAIN HIGH  (multi-factor reasoning)
        2. Single strong triggers     → HIGH
        3. Soft / channel signals     → MEDIUM RISK (REVIEW)
        4. Default                    → LOW RISK (APPROVED)
    """
    is_unverified = "unverified" in device.lower()
    ip_high = ip_status in HIGH_RISK_IPS
    ip_medium = ip_status in MEDIUM_RISK_IPS
    unverified_over_thresh = is_unverified and amount > UNVERIFIED_DEVICE_THRESHOLD
    telemetry_gap = (
        device == "Unknown"
        and ip_status == "Unspecified"
        and amount > TELEMETRY_GAP_THRESHOLD
    )
    is_crypto = method == "Crypto"
    is_wire = method == "Wire Transfer"

    # ── 1. Compound triggers → CERTAIN HIGH ──────────────────────────────────

    if ip_high and unverified_over_thresh:
        risk_path = "compound_ip_unverified"
        reason = (
            f"Multiple risk factors compound this assessment: suspicious network "
            f"origin ({ip_status}) combined with an unverified device ({device}) "
            f"executing a transaction of ${amount:.2f} above the "
            f"${UNVERIFIED_DEVICE_THRESHOLD:.2f} unverified-device threshold."
        )
        return "certain", risk_path, VERDICT_FOR_BAND["certain"], reason

    if ip_high and is_crypto:
        risk_path = "compound_ip_crypto"
        reason = (
            f"Multiple risk factors compound this assessment: suspicious network "
            f"origin ({ip_status}) combined with a Crypto payment channel — a "
            f"high-anonymity method — for a transaction of ${amount:.2f}."
        )
        return "certain", risk_path, VERDICT_FOR_BAND["certain"], reason

    # ── 2. Single strong triggers → HIGH ─────────────────────────────────────

    if ip_high:
        risk_path = "ip_high_single"
        method_note = (
            " Wire Transfer payment channel adds additional traceability concern."
            if is_wire
            else ""
        )
        reason = (
            f"High risk identified due to suspicious network origin "
            f"({ip_status}).{method_note}"
        )
        return "high", risk_path, VERDICT_FOR_BAND["high"], reason

    if unverified_over_thresh:
        risk_path = "unverified_high_single"
        method_note = (
            " Wire Transfer payment channel adds traceability scrutiny."
            if is_wire
            else " Crypto payment channel adds additional anonymity concern."
            if is_crypto
            else ""
        )
        reason = (
            f"High transaction amount (${amount:.2f}) exceeds the unverified-device "
            f"threshold of ${UNVERIFIED_DEVICE_THRESHOLD:.2f} executed from an "
            f"unverified device ({device}).{method_note}"
        )
        return "high", risk_path, VERDICT_FOR_BAND["high"], reason

    if telemetry_gap:
        risk_path = "missing_telemetry_high"
        reason = (
            f"High transaction amount (${amount:.2f}) exceeds the telemetry gap "
            f"threshold of ${TELEMETRY_GAP_THRESHOLD:.2f} with missing device and "
            f"network metadata, requiring mandatory manual verification."
        )
        return "high", risk_path, VERDICT_FOR_BAND["high"], reason

    # VPN/Proxy + Crypto → combined anonymity escalates to HIGH
    if ip_medium and is_crypto and amount > VPN_MEDIUM_AMOUNT_THRESHOLD:
        risk_path = "vpn_crypto_elevated_high"
        reason = (
            f"Elevated risk: transaction of ${amount:.2f} via Crypto payment "
            f"channel on an anonymising network ({ip_status}). The combined "
            f"anonymity of Crypto and {ip_status} escalates this beyond the "
            f"standard manual-review threshold."
        )
        return "high", risk_path, VERDICT_FOR_BAND["high"], reason

    # ── 3. Soft / channel signals → MEDIUM RISK ───────────────────────────────

    if ip_medium and amount > VPN_MEDIUM_AMOUNT_THRESHOLD:
        risk_path = "vpn_medium"
        reason = (
            f"Moderate concern: {method} transaction of ${amount:.2f} on an "
            f"anonymising network ({ip_status}). Amount falls below the automatic "
            f"flag threshold but the network origin warrants manual review."
        )
        return "medium", risk_path, VERDICT_FOR_BAND["medium"], reason

    if is_wire and amount > WIRE_MEDIUM_AMOUNT_THRESHOLD and not is_unverified:
        risk_path = "wire_medium"
        reason = (
            f"Moderate concern: Wire Transfer of ${amount:.2f} exceeds the "
            f"${WIRE_MEDIUM_AMOUNT_THRESHOLD:.2f} elevated-scrutiny threshold "
            f"for this payment channel. Device and network parameters are within "
            f"normal range."
        )
        return "medium", risk_path, VERDICT_FOR_BAND["medium"], reason

    if is_crypto and amount > CRYPTO_MEDIUM_AMOUNT_THRESHOLD:
        risk_path = "crypto_medium"
        reason = (
            f"Moderate concern: Crypto payment of ${amount:.2f} exceeds the "
            f"${CRYPTO_MEDIUM_AMOUNT_THRESHOLD:.2f} elevated-scrutiny threshold "
            f"for high-anonymity payment channels. Network and device parameters "
            f"are within normal range."
        )
        return "medium", risk_path, VERDICT_FOR_BAND["medium"], reason

    # ── 4. Default low risk ───────────────────────────────────────────────────

    if device == "Unknown" and ip_status == "Unspecified":
        risk_path = "missing_telemetry_low"
        reason = (
            f"Low financial exposure (${amount:.2f}) falls below the "
            f"${TELEMETRY_GAP_THRESHOLD:.2f} telemetry-gap threshold, "
            f"mitigating the missing metadata risk."
        )
    else:
        risk_path = "standard_low"
        reason = "Parameters fall within standard safe operational thresholds."

    return "low", risk_path, VERDICT_FOR_BAND["low"], reason


# ─────────────────────────────────────────────────────────────────────────────
# Risk evaluation — in-app chat  [FIX: CRITICAL-2, MODERATE-2, MINOR-1/2]
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_chat_risk(
    ip_status: str,
    device: str,
    has_malicious_payload: bool,
) -> tuple[str, str, str, str, str]:
    """
    Evaluates in-app chat message risk.

    Returns (band, risk_path, verdict, payload_desc, reason).

    Priority: compound → high → medium → low.
    """
    ip_high = ip_status in HIGH_RISK_IPS
    ip_medium = ip_status in MEDIUM_RISK_IPS

    # Compound: malicious payload + high-risk IP → CERTAIN HIGH
    if has_malicious_payload and ip_high:
        risk_path = "chat_compound_payload_ip"
        payload_desc = (
            "Flagged text payload for social engineering threat signals "
            "(phishing link / urgency lure)."
        )
        reason = (
            f"Multiple risk factors compound this assessment: malicious payload "
            f"content combined with suspicious network origin ({ip_status})."
        )
        return "certain", risk_path, VERDICT_FOR_BAND["certain"], payload_desc, reason

    # Malicious payload alone → HIGH
    if has_malicious_payload:
        risk_path = "chat_malicious_payload"
        payload_desc = (
            "Flagged text payload for social engineering threat signals "
            "(phishing link / urgency lure)."
        )
        reason = "Malicious payload content overrides device and IP status attributes."
        return "high", risk_path, VERDICT_FOR_BAND["high"], payload_desc, reason

    # High-risk IP + benign payload → HIGH
    if ip_high:
        risk_path = "chat_ip_high"
        payload_desc = (
            "Text payload contains standard conversational prose with zero "
            "threat indicators."
        )
        reason = (
            f"Benign message payload, but high risk identified due to "
            f"suspicious network origin ({ip_status})."
        )
        return "high", risk_path, VERDICT_FOR_BAND["high"], payload_desc, reason

    # Medium-risk IP + benign payload → MEDIUM  [FIX: MODERATE-2]
    if ip_medium:
        risk_path = "chat_ip_medium"
        payload_desc = (
            "Text payload contains standard conversational prose with zero "
            "threat indicators."
        )
        reason = (
            f"Benign message payload, but moderate concern identified due to "
            f"anonymising network origin ({ip_status}). Manual review recommended."
        )
        return "medium", risk_path, VERDICT_FOR_BAND["medium"], payload_desc, reason

    # Default: clean profile + benign payload → LOW
    risk_path = "chat_low"
    payload_desc = (
        "Text payload contains standard conversational prose with zero "
        "threat indicators."
    )
    reason = "Message content and device telemetry show no threat indicators."
    return "low", risk_path, VERDICT_FOR_BAND["low"], payload_desc, reason


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic record generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_v5_record(idx: int) -> dict:
    """
    Synthesises one V5-format record. Scores are now fully deterministic per
    semantic risk profile — identical profiles always yield the same score.
    """
    user_id = f"U-{1000000 + idx}"

    omit_telemetry = random.random() < 0.10
    device = random.choice(DEVICE_TYPES) if not omit_telemetry else "Unknown"
    ip_status = random.choice(IP_TYPES) if not omit_telemetry else "Unspecified"

    is_chat = random.random() < 0.35

    if is_chat:
        has_malicious_payload = random.random() < 0.40
        payload = (
            random.choice(MALICIOUS_PAYLOADS)
            if has_malicious_payload
            else random.choice(BENIGN_PAYLOADS)
        )

        band, risk_path, verdict, payload_desc, reason = evaluate_chat_risk(
            ip_status, device, has_malicious_payload
        )
        risk_score = score_for_band(band, device, ip_status, risk_path)

        prompt = (
            f"User {user_id} attempting message action via In-App Chat. "
            f"Device: {device}. IP Status: {ip_status}. Payload: {payload}"
        )
        response = (
            f"Step 1: Context Analysis - Evaluated user {user_id} attempting"
            f" message action via In-App Chat using {device}.\n"
            f"Step 2: Payload Analysis - {payload_desc}\n"
            f"Step 3: Risk Scoring - Assigned risk score of {risk_score} based"
            f" on context and payload evaluation.\n"
            f"Step 4: Synthesis & Reasoning - {reason}"
            f" Final Assessment: {verdict}."
        )

    else:
        amount = round(random.uniform(10.0, 5000.0), 2)
        method = random.choice(PAYMENT_METHODS)

        band, risk_path, verdict, reason = evaluate_payment_risk(
            amount, method, device, ip_status
        )
        risk_score = score_for_band(band, device, ip_status, risk_path)

        prompt = (
            f"User {user_id} attempting payment of ${amount:.2f} via {method}."
            f" Device: {device}. IP Status: {ip_status}."
        )
        response = (
            f"Step 1: Context Analysis - Evaluated user {user_id} requesting"
            f" ${amount:.2f} transaction via {method} using {device}.\n"
            f"Step 2: Risk Scoring - Assigned transaction risk score of"
            f" {risk_score} based on metadata parameters.\n"
            f"Step 3: Synthesis & Reasoning - IP status is confirmed as"
            f" {ip_status}. {reason} Final Assessment: {verdict}."
        )

    return {
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Targeted MEDIUM generator  [FIX: EVAL-1]
# ─────────────────────────────────────────────────────────────────────────────


def generate_medium_record(idx: int) -> dict:
    """
    Synthesises one record guaranteed to land in the MEDIUM RISK band.
    Cycles through MEDIUM_TRIGGER_CYCLE so every soft-signal path
    (VPN payment, Wire, Crypto, chat-VPN, Unknown Proxy) is evenly covered.

    All inputs are chosen to satisfy the MEDIUM conditions in
    evaluate_payment_risk() / evaluate_chat_risk() — no post-hoc filtering
    needed and no risk of accidentally producing a HIGH or LOW record.
    """
    user_id = f"U-{2000000 + idx}"
    trigger = MEDIUM_TRIGGER_CYCLE[idx % len(MEDIUM_TRIGGER_CYCLE)]

    if trigger == "chat_vpn_benign":
        ip_status = random.choice(list(MEDIUM_RISK_IPS))
        device = random.choice(DEVICE_TYPES)
        payload = random.choice(BENIGN_PAYLOADS)
        band, risk_path, verdict, payload_desc, reason = evaluate_chat_risk(
            ip_status, device, has_malicious_payload=False
        )
        risk_score = score_for_band(band, device, ip_status, risk_path)
        prompt = (
            f"User {user_id} attempting message action via In-App Chat. "
            f"Device: {device}. IP Status: {ip_status}. Payload: {payload}"
        )
        response = (
            f"Step 1: Context Analysis - Evaluated user {user_id} attempting"
            f" message action via In-App Chat using {device}.\n"
            f"Step 2: Payload Analysis - {payload_desc}\n"
            f"Step 3: Risk Scoring - Assigned risk score of {risk_score} based"
            f" on context and payload evaluation.\n"
            f"Step 4: Synthesis & Reasoning - {reason}"
            f" Final Assessment: {verdict}."
        )
    else:
        # Payment scenarios ────────────────────────────────────────────────────
        if trigger == "vpn_payment":
            ip_status = random.choice(list(MEDIUM_RISK_IPS))
            device = random.choice(_VERIFIED_DEVICES)
            # Exclude Crypto — VPN + Crypto above threshold escalates to HIGH
            method = random.choice(["Credit Card", "Wire Transfer", "PayPal"])
            amount = round(random.uniform(VPN_MEDIUM_AMOUNT_THRESHOLD + 1.0, 4990.0), 2)

        elif trigger == "wire_clean_payment":
            ip_status = random.choice(["Clean IP", "Residential ISP"])
            device = random.choice(_VERIFIED_DEVICES)
            method = "Wire Transfer"
            amount = round(
                random.uniform(WIRE_MEDIUM_AMOUNT_THRESHOLD + 1.0, 4990.0), 2
            )

        elif trigger == "crypto_clean_payment":
            ip_status = random.choice(["Clean IP", "Residential ISP"])
            device = random.choice(_VERIFIED_DEVICES)
            method = "Crypto"
            amount = round(
                random.uniform(CRYPTO_MEDIUM_AMOUNT_THRESHOLD + 1.0, 4990.0), 2
            )

        else:  # unknown_proxy_payment
            ip_status = "Unknown Proxy"
            device = random.choice(_VERIFIED_DEVICES)
            # Exclude Crypto — Unknown Proxy + Crypto above threshold → HIGH
            method = random.choice(["Credit Card", "PayPal"])
            amount = round(random.uniform(VPN_MEDIUM_AMOUNT_THRESHOLD + 1.0, 4990.0), 2)

        band, risk_path, verdict, reason = evaluate_payment_risk(
            amount, method, device, ip_status
        )
        risk_score = score_for_band(band, device, ip_status, risk_path)
        prompt = (
            f"User {user_id} attempting payment of ${amount:.2f} via {method}."
            f" Device: {device}. IP Status: {ip_status}."
        )
        response = (
            f"Step 1: Context Analysis - Evaluated user {user_id} requesting"
            f" ${amount:.2f} transaction via {method} using {device}.\n"
            f"Step 2: Risk Scoring - Assigned transaction risk score of"
            f" {risk_score} based on metadata parameters.\n"
            f"Step 3: Synthesis & Reasoning - IP status is confirmed as"
            f" {ip_status}. {reason} Final Assessment: {verdict}."
        )

    return {
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Boundary LOW generator  [FIX: EVAL-2]
# ─────────────────────────────────────────────────────────────────────────────


def generate_boundary_record(idx: int) -> dict:
    """
    Synthesises a LOW RISK record with an amount just below a MEDIUM threshold.

    These hard-negative contrast examples teach the model exactly where each
    MEDIUM / LOW boundary sits, addressing the MED→LOW (37) and LOW→MED (26)
    confusion in the V6 eval.  Three boundary types rotate by index:

      0 — VPN/Proxy + amount just below $1,500  → LOW  (not enough for vpn_medium)
      1 — Wire Transfer + amount just below $2,000 → LOW  (not enough for wire_medium)
      2 — Crypto + clean IP + amount just below $1,000 → LOW (not enough for crypto_medium)
    """
    user_id = f"U-{3000000 + idx}"
    boundary_type = idx % 3

    if boundary_type == 0:
        ip_status = random.choice(list(MEDIUM_RISK_IPS))
        device = random.choice(DEVICE_TYPES)
        method = random.choice(["Credit Card", "PayPal"])
        amount = round(
            random.uniform(
                VPN_MEDIUM_AMOUNT_THRESHOLD * 0.5, VPN_MEDIUM_AMOUNT_THRESHOLD - 0.01
            ),
            2,
        )
    elif boundary_type == 1:
        ip_status = random.choice(["Clean IP", "Residential ISP"])
        device = random.choice(_VERIFIED_DEVICES)
        method = "Wire Transfer"
        amount = round(
            random.uniform(
                WIRE_MEDIUM_AMOUNT_THRESHOLD * 0.5, WIRE_MEDIUM_AMOUNT_THRESHOLD - 0.01
            ),
            2,
        )
    else:
        ip_status = random.choice(["Clean IP", "Residential ISP"])
        device = random.choice(_VERIFIED_DEVICES)
        method = "Crypto"
        amount = round(
            random.uniform(
                CRYPTO_MEDIUM_AMOUNT_THRESHOLD * 0.3,
                CRYPTO_MEDIUM_AMOUNT_THRESHOLD - 0.01,
            ),
            2,
        )

    band, risk_path, verdict, reason = evaluate_payment_risk(
        amount, method, device, ip_status
    )
    risk_score = score_for_band(band, device, ip_status, risk_path)

    prompt = (
        f"User {user_id} attempting payment of ${amount:.2f} via {method}."
        f" Device: {device}. IP Status: {ip_status}."
    )
    response = (
        f"Step 1: Context Analysis - Evaluated user {user_id} requesting"
        f" ${amount:.2f} transaction via {method} using {device}.\n"
        f"Step 2: Risk Scoring - Assigned transaction risk score of"
        f" {risk_score} based on metadata parameters.\n"
        f"Step 3: Synthesis & Reasoning - IP status is confirmed as"
        f" {ip_status}. {reason} Final Assessment: {verdict}."
    )
    return {
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# V4 ingestion helpers
# ─────────────────────────────────────────────────────────────────────────────


def extract_prompt_and_response(record: dict) -> tuple[str, str]:
    """Extracts prompt and response from ChatML or legacy flat-format records."""
    if "messages" in record and isinstance(record["messages"], list):
        user_msg = ""
        assistant_msg = ""
        for msg in record["messages"]:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
        return user_msg, assistant_msg
    return record.get("prompt", ""), record.get("response", "")


def safe_json_loads(line: str) -> dict | None:
    """Parses a JSONL line, attempting one-shot recovery for truncated records."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        if not line.endswith("}"):
            try:
                return json.loads(line + "}")
            except json.JSONDecodeError:
                pass
        return None


def validate_and_normalize_v4(record: dict) -> dict | None:
    """
    Validates, cleans, and standardises a V4 record into ChatML format.

    Rejection checks (any one triggers a drop):
        1. Phantom attribute hallucination in response not present in prompt.
        2. IP type mentioned in response that wasn't in the prompt.
        3. Clean IP in prompt paired with high-risk + network-origin reasoning.
        4. Unverified-device HIGH RISK below $3,000 without a flagged IP —
           directly contradicts the V5 rule.  [FIX: MODERATE-1]

    Accepted records are step-format normalised and emitted as ChatML.
    """
    if not isinstance(record, dict):
        return None

    prompt, response = extract_prompt_and_response(record)
    if not prompt or not response:
        return None

    p_lower = prompt.lower()
    r_lower = response.lower()

    # ── Check 1: Phantom attribute hallucinations ─────────────────────────────
    for pattern in PHANTOM_PATTERNS:
        if re.search(pattern, r_lower) and not re.search(pattern, p_lower):
            return None

    # ── Check 2: Cross-contaminated IP classifications ────────────────────────
    ip_terms = [
        "blacklisted ip",
        "tor exit node",
        "vpn datacenter",
        "unknown proxy",
        "clean ip",
        "residential isp",
        "unspecified",
    ]
    prompt_ips = [ip for ip in ip_terms if ip in p_lower]
    if prompt_ips:
        for ip in ip_terms:
            if ip in r_lower and ip not in prompt_ips:
                return None

    # ── Check 3: Contradictory Clean IP + high-risk + network-origin reasoning ─
    if "clean ip" in p_lower and "high risk" in r_lower and "network origin" in r_lower:
        return None

    # ── Check 4: V4 ↔ V5 unverified-device rule conflict  [FIX: MODERATE-1] ──
    #   V5 rule: unverified device + amount ≤ $3,000 + no flagged IP → LOW RISK
    #   Reject V4 records that mark this scenario HIGH RISK.
    if "unverified" in p_lower and "high risk" in r_lower:
        amount_match = re.search(r"\$(\d+\.?\d*)", prompt)
        if amount_match:
            try:
                amount = float(amount_match.group(1))
                ip_flagged = any(
                    ip in p_lower for ip in ["blacklisted ip", "tor exit node"]
                )
                if amount <= UNVERIFIED_DEVICE_THRESHOLD and not ip_flagged:
                    return (
                        None  # contradicts V5 rule — drop to prevent training conflict
                    )
            except ValueError:
                pass

    # ── Check 5: Wrong Step 4 label on chat records  [FIX: EVAL-4] ───────────
    #   35 eval records had "Step 4: Final Assessment" — a label that drifted
    #   from the canonical "Step 4: Synthesis & Reasoning". Drop at ingestion.
    if ("in-app chat" in p_lower or "payload" in p_lower) and "step 4" in r_lower:
        if not re.search(r"Step 4:\s*Synthesis & Reasoning", response, re.IGNORECASE):
            return None

    # ── Check 6: Bullet points in response  [FIX: EVAL-5] ────────────────────
    #   14 eval records had bullet-point payload analysis (e.g. "- Urgency lure").
    #   Plain prose is the required format; any markdown list marker rejects.
    if re.search(r"^\s*[-•*]\s+\S", response, re.MULTILINE):
        return None

    # ── Check 7: Score 0.00 or out of range  [FIX: EVAL-3] ───────────────────
    score_m = re.search(r"risk score of ([\d.]+)", response)
    if score_m:
        try:
            if not (0.01 <= float(score_m.group(1)) <= 0.99):
                return None
        except ValueError:
            return None

    # ── Check 8: Hallucinated threshold values  [FIX: EVAL-6] ───────────────
    #   4 eval records cited thresholds like $300 that don't exist in our rules.
    #   Any threshold dollar amount must match one of the five known constants.
    for m in re.finditer(
        r"\$([\d,]+(?:\.\d+)?)\s+(?:[\w-]+\s+)?threshold"
        r"|threshold\s+of\s+\$([\d,]+(?:\.\d+)?)",
        response,
        re.IGNORECASE,
    ):
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        if raw:
            try:
                val = float(raw)
                if not any(abs(val - t) < 0.01 for t in _VALID_THRESHOLD_AMOUNTS):
                    return None
            except ValueError:
                return None

    # ── Check 9: Malformed verdict string  [FIX: EVAL-7] ─────────────────────
    #   Records with "Final Assessment: FLAGGED." instead of the canonical form
    #   corrupt the model's output vocabulary. Drop them.
    verdict_m = re.search(r"Final Assessment:\s*(.+?)\.", response)
    if verdict_m and verdict_m.group(1).strip() not in _CANONICAL_VERDICTS:
        return None

    # ── Step-format normalisation ─────────────────────────────────────────────
    normalized_resp = re.sub(
        r"Step 1\s*[-:]\s*(Context Analysis|Context)?:?\s*",
        "Step 1: Context Analysis - ",
        response,
        flags=re.IGNORECASE,
    )

    if "payload analysis" in r_lower or "payload:" in p_lower:
        normalized_resp = re.sub(
            r"Step 2\s*[-:]\s*(Payload Analysis|Payload)?:?\s*",
            "\nStep 2: Payload Analysis - ",
            normalized_resp,
            flags=re.IGNORECASE,
        )
        normalized_resp = re.sub(
            r"Step 3\s*[-:]\s*(Risk Scoring|Scoring)?:?\s*",
            "\nStep 3: Risk Scoring - ",
            normalized_resp,
            flags=re.IGNORECASE,
        )
        normalized_resp = re.sub(
            r"Step 4\s*[-:]\s*(Synthesis & Reasoning|Reasoning)?:?\s*",
            "\nStep 4: Synthesis & Reasoning - ",
            normalized_resp,
            flags=re.IGNORECASE,
        )
    else:
        normalized_resp = re.sub(
            r"Step 2\s*[-:]\s*(Risk Scoring|Scoring)?:?\s*",
            "\nStep 2: Risk Scoring - ",
            normalized_resp,
            flags=re.IGNORECASE,
        )
        normalized_resp = re.sub(
            r"Step 3\s*[-:]\s*(Synthesis & Reasoning|Reasoning)?:?\s*",
            "\nStep 3: Synthesis & Reasoning - ",
            normalized_resp,
            flags=re.IGNORECASE,
        )

    return {
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": normalized_resp.strip()},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def get_verdict_label(record: dict) -> str:
    """
    Extracts the verdict class label from a record's assistant message.
    Used as the stratification key for the train/val split.
    Returns one of: 'high', 'medium', 'low'.
    """
    content = record["messages"][2]["content"]
    if "HIGH RISK" in content:
        return "high"
    if "MEDIUM RISK" in content:
        return "medium"
    return "low"


def stratified_split(
    records: list[dict],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Splits records into train/val while preserving the class distribution
    of each verdict label (high / medium / low) within ±1 record per class.

    This replaces the naive random slice that caused a 5.5% HIGH-class skew
    between train (48.2%) and val (53.7%) in the previous build.
    """
    rng = random.Random(random_seed)

    # Group records by verdict label
    buckets: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
    for rec in records:
        buckets[get_verdict_label(rec)].append(rec)

    train_records: list[dict] = []
    val_records: list[dict] = []

    for label, group in buckets.items():
        rng.shuffle(group)
        n_val = round(len(group) * val_ratio)
        val_records.extend(group[:n_val])
        train_records.extend(group[n_val:])

    # Shuffle both splits so classes are interleaved, not block-sorted
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    return train_records, val_records


def _print_split_report(
    train: list[dict],
    val: list[dict],
    train_path: str,
    val_path: str,
) -> None:
    """Prints a per-class distribution table for both splits."""
    labels = ["high", "medium", "low"]
    tags = {
        "high": "HIGH RISK (FLAGGED)",
        "medium": "MEDIUM RISK (REVIEW)",
        "low": "LOW RISK (APPROVED)",
    }
    print("\nDataset Curation Complete:")
    print(
        f"  {'Label':<8}  {'Train':>6}  {'Train%':>7}  {'Val':>6}  {'Val%':>7}  {'Delta':>7}"
    )
    print(f"  {'-' * 8}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 7}  {'-' * 7}")
    for lbl in labels:
        tag = tags[lbl]
        tr_n = sum(1 for r in train if tag in r["messages"][2]["content"])
        vl_n = sum(1 for r in val if tag in r["messages"][2]["content"])
        tr_p = tr_n / len(train) * 100
        vl_p = vl_n / len(val) * 100
        delta = abs(tr_p - vl_p)
        flag = "  ✓" if delta <= 1.5 else f"  ← {delta:.1f}% skew"
        print(
            f"  {lbl.upper():<8}  {tr_n:>6}  {tr_p:>6.1f}%  {vl_n:>6}  {vl_p:>6.1f}%  {delta:>6.1f}%{flag}"
        )
    print(f"\n  Train : {len(train):>5} records → '{train_path}'")
    print(f"  Val   : {len(val):>5} records → '{val_path}'")


def build_v5_pipeline(
    v4_input_path: str,
    train_out_path: str = "train_2400.jsonl",
    val_out_path: str = "val_600.jsonl",
    total_target: int = 3000,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> None:
    """
    Full curation pipeline — V4 ingestion → targeted synthesis → format
    validation → stratified 80/20 split.

    Synthesis strategy  [FIX: EVAL-1, EVAL-2]:
      1. MEDIUM records  — generate_medium_record() cycles through all five
         soft-signal trigger types, guaranteeing MEDIUM output every call.
         Fills to TARGET_MEDIUM_RATIO of total_target.
      2. Boundary records — generate_boundary_record() produces LOW records
         with amounts just below each MEDIUM threshold (hard negatives).
         Fills to BOUNDARY_RATIO of total_target.
      3. General records — generate_v5_record() fills the remaining quota
         (naturally weighted toward HIGH + LOW by probability).

    Every record — ingested or synthesised — passes through validate_format()
    before being admitted to the dataset.  [FIX: EVAL-3 through EVAL-7]
    """
    v5_records: list[dict] = []
    dropped_count: int = 0
    format_dropped: int = 0

    # ── Step 1: V4 ingestion ──────────────────────────────────────────────────
    try:
        with open(v4_input_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = safe_json_loads(line)
                if parsed:
                    validated = validate_and_normalize_v4(parsed)
                    if validated:
                        if validate_format(validated):
                            v5_records.append(validated)
                        else:
                            format_dropped += 1
                            dropped_count += 1
                    else:
                        dropped_count += 1
                else:
                    dropped_count += 1
    except FileNotFoundError:
        print(
            f"File '{v4_input_path}' not found. Synthesising full dataset from scratch."
        )

    print(
        f"Ingestion Report: Retained {len(v5_records)} valid records | "
        f"Dropped {dropped_count} (of which {format_dropped} failed format validation)."
    )

    # ── Step 2: Targeted synthesis ────────────────────────────────────────────
    if len(v5_records) < total_target:
        current = {
            cls: sum(1 for r in v5_records if get_verdict_label(r) == cls)
            for cls in ("high", "medium", "low")
        }
        target_medium = round(total_target * TARGET_MEDIUM_RATIO)
        target_boundary = round(total_target * BOUNDARY_RATIO)

        # 2a. MEDIUM — guaranteed via dedicated generator
        medium_needed = max(0, target_medium - current["medium"])
        print(f"Synthesising {medium_needed} targeted MEDIUM records...")
        med_added = 0
        for i in range(medium_needed * 3):  # 3× budget absorbs format rejects
            if med_added >= medium_needed:
                break
            rec = generate_medium_record(i)
            if validate_format(rec):
                v5_records.append(rec)
                med_added += 1

        # 2b. Boundary LOW — hard negatives at each MEDIUM threshold
        print(f"Synthesising {target_boundary} boundary LOW records...")
        bnd_added = 0
        for i in range(target_boundary * 3):
            if bnd_added >= target_boundary:
                break
            rec = generate_boundary_record(i)
            if validate_format(rec) and get_verdict_label(rec) == "low":
                v5_records.append(rec)
                bnd_added += 1

        # 2c. General synthesis — fills remaining quota with a class-aware gate.
        # Without the gate, the general generator would also produce ~22% MEDIUM
        # records, pushing the total MEDIUM count well above TARGET_MEDIUM_RATIO.
        # The gate rejects any record whose class has already reached its target.
        target_counts = {
            "high": round(total_target * TARGET_HIGH_RATIO),
            "medium": round(total_target * TARGET_MEDIUM_RATIO),
            "low": round(total_target * TARGET_LOW_RATIO),
        }
        live_counts = {
            cls: sum(1 for r in v5_records if get_verdict_label(r) == cls)
            for cls in ("high", "medium", "low")
        }
        still_needed = total_target - len(v5_records)
        print(f"Synthesising {still_needed} general records (class-gated)...")
        gen_idx = gen_added = 0
        while gen_added < still_needed and gen_idx < still_needed * 8:
            rec = generate_v5_record(gen_idx)
            gen_idx += 1
            if not validate_format(rec):
                continue
            cls = get_verdict_label(rec)
            if live_counts[cls] >= target_counts[cls]:
                continue  # this class is full — skip the record
            v5_records.append(rec)
            live_counts[cls] += 1
            gen_added += 1

    # ── Step 3: Final format pass + trim ─────────────────────────────────────
    before = len(v5_records)
    v5_records = [r for r in v5_records if validate_format(r)]
    if len(v5_records) < before:
        print(
            f"Format pass: removed {before - len(v5_records)} non-conforming records."
        )

    v5_records = v5_records[:total_target]

    # ── Step 4: Stratified split ──────────────────────────────────────────────
    train_records, val_records = stratified_split(
        v5_records, val_ratio=val_ratio, random_seed=random_seed
    )

    # ── Step 5: Write ─────────────────────────────────────────────────────────
    with open(train_out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in train_records)
    with open(val_out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in val_records)

    _print_split_report(
        train_records, val_records, train_out_path, val_path=val_out_path
    )


if __name__ == "__main__":
    build_v5_pipeline("fraud_detection_dataset_V4.jsonl")

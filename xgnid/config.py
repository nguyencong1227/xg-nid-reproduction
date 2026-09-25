from __future__ import annotations

# --- Component 1: Flow and Feature Generator (Sec. 3.1.1) --------------------
# "we set a maximum limit of 20 packets per flow"
MAX_PACKETS_PER_FLOW = 20
# "we set an idle timeout of 120 seconds"
IDLE_TIMEOUT_S = 120
# NFStream default; flows are cut by MAX_PACKETS_PER_FLOW long before this.
ACTIVE_TIMEOUT_S = 1800
# accounting_mode=1 -> IP layer accounting (matches the released GNN4ID tool).
ACCOUNTING_MODE = 1
# "the payload of each packet is represented by 1500 features derived from the
#  bytes of the payload"
PAYLOAD_DIM = 1500

# The 14 packet-level features attached to every flow record.
PACKET_FEATURE_COLUMNS = [
    "udps.payload_data",
    "udps.delta_time",
    "udps.packet_direction",
    "udps.ip_size",
    "udps.transport_size",
    "udps.payload_size",
    "udps.syn",
    "udps.cwr",
    "udps.ece",
    "udps.urg",
    "udps.ack",
    "udps.psh",
    "udps.rst",
    "udps.fin",
]

# --- Component 2: Explainable Feature Extractor (Sec. 3.1.2, Table 1) --------
# The released GNN4ID tool uses a 350-*row* rolling window per destination.
ROLLING_WINDOW = 350
HTTP_PORTS = [80, 443, 8080]
DNS_PORTS = [53]
VULNERABLE_PORTS = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080]

# --- Categorical vocabularies (fixed so every shard yields identical columns)
# CIC-IoT2023 only contains these five L4/L3 protocol numbers.
PROTOCOLS = [1, 2, 6, 17, 58]          # ICMP, IGMP, TCP, UDP, ICMPv6
EXPIRATION_IDS = [0, -1]               # idle/active expiry vs. forced (20-packet) expiry

# --- Dataset: CIC-IoT2023 (Sec. 3.2) ----------------------------------------
LABEL_DICT = {
    "Benign": 0,
    "WebBased": 1,
    "Spoofing": 2,
    "Recon": 3,
    "Mirai": 4,
    "DoS": 5,
    "DDoS": 6,
    "BruteForce": 7,
}
CLASS_NAMES = [name for name, _ in sorted(LABEL_DICT.items(), key=lambda kv: kv[1])]

# Attacks whose signal lives in the payload; Algorithm 3 line 18 triggers the
# second (payload) LLM query only for these.
PAYLOAD_SPECIFIC_CLASSES = {"WebBased", "BruteForce"}

# Table 3: flows must touch one of these MACs to count as an attack, and must
# not touch any of them to count as benign.
ATTACKER_MACS = {
    "e4:5f:01:55:90:c4",
    "dc:a6:32:c9:e4:d5",
    "dc:a6:32:dc:27:d5",
    "dc:a6:32:c9:e5:ef",
    "dc:a6:32:c9:e4:ab",
    "dc:a6:32:c9:e4:90",
    "dc:a6:32:c9:e5:a4",
    "b0:09:da:3e:82:6c",
    "ac:17:02:05:34:27",
}

# Maps a CIC-IoT2023 pcap basename (33 attack captures + benign) to its class.
PCAP_PREFIX_TO_CLASS = {
    # --- DDoS (12) ---
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    # --- DoS (4) ---
    "DoS-UDP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # --- Mirai (3) ---
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # --- Recon (5) ---
    "Recon-PingSweep": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    # --- Spoofing (2) ---
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # --- Web-based (6) ---
    "BrowserHijacking": "WebBased",
    "Backdoor_Malware": "WebBased",
    "XSS": "WebBased",
    "Uploading_Attack": "WebBased",
    "SqlInjection": "WebBased",
    "CommandInjection": "WebBased",
    # --- Brute force (1) ---
    "DictionaryBruteForce": "BruteForce",
    # --- Benign ---
    "Benign": "Benign",
    "BenignTraffic": "Benign",
}


def class_from_pcap_name(name: str) -> str | None:
    """Resolve a CIC-IoT2023 pcap file name to one of the eight classes.

    Capture files are chunked (``XSS1.pcap``, ``DDoS-TCP_Flood23.pcap``), so the
    longest matching prefix wins.
    """
    import os
    import re

    stem = os.path.basename(name)
    for suffix in (".pcap", ".pcapng"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # strip a trailing chunk index and any separator left behind
    stem = re.sub(r"[-_]?\d+$", "", stem)
    for prefix in sorted(PCAP_PREFIX_TO_CLASS, key=len, reverse=True):
        if stem.startswith(prefix):
            return PCAP_PREFIX_TO_CLASS[prefix]
    return None

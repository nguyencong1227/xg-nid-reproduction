"""Component 6 - Generative Explainer (XG-NID Sec. 3.1.6, Algorithm 3).

Turns the Integrated Gradients output into the two zero-shot prompts the paper
specifies and, optionally, runs them through an LLM.

* ``Q_flow``    = P_init + P_part2 + P_align
* ``Q_payload`` = P_payloadPrefix + P_ascii + P_align, emitted only when the
  predicted class is payload-specific (web-based or brute force).

The wording of P_init, P_part2, P_payloadPrefix and P_align is taken verbatim
from Sec. 3.1.6 so the prompts are reproducible.  ``FEATURE_DESCRIPTIONS``
supplies N_feat, the human-readable feature names of Algorithm 3 line 6 --
without it the LLM would be reasoning about strings like
``Rolling_SYN_Packets_Destination``.
"""
from __future__ import annotations

import re

P_INIT = "The predicted class from GNN is {predicted_class}."
P_PART2_HEADER = "The top features contributing to this prediction are:"
P_ALIGN = (
    "Don't expect any values on your own. Explain the predicted outcome and its "
    "potential reason along with the potential mitigation. "
    'Start your answer with "The predicted outcome is.'
)
P_PAYLOAD_PREFIX = (
    "Analyze whether this payload of network flow is malicious or not. "
    "Give reason concisely."
)

# N_feat: descriptive names for the Table 1 temporal features and the NFStream
# statistics that survive into the flow node.
FEATURE_DESCRIPTIONS = {
    "Rolling_UDP_Requests": "cumulative count of UDP flows",
    "Rolling_TCP_Requests": "cumulative count of TCP flows",
    "Rolling_ICMP_Requests": "cumulative count of ICMP requests",
    "Rolling_ACK_Packets": "cumulative count of TCP ACK packets",
    "Rolling_FIN_Packets": "cumulative count of TCP FIN packets",
    "Rolling_rst_Packets": "cumulative count of TCP RST packets",
    "Rolling_psh_Packets": "cumulative count of TCP PSH packets",
    "Rolling_SYN_Packets": "cumulative count of TCP SYN packets",
    "Rolling_http_port": "count of access attempts to well-known HTTP ports",
    "Rolling_DNS_request": "cumulative count of DNS requests",
    "Rolling_DNS_request2": "cumulative count of DNS responses",
    "Rolling_vulnerable_port": "count of contacts to known vulnerable ports",
    "Rolling_Duration": "average duration of bidirectional sessions (ms)",
    "Rolling_packets": "cumulative count of packets sent to the destination",
    "Rolling_bipackets": "cumulative count of bidirectional packets",
    "Unique_Ports_In_SourceDestination":
        "number of unique destination ports contacted by this source",
    "packet_size_variation": "spread of packet sizes across both directions",
}

_SCOPE = {
    "_Destination": " at the destination within the rolling window",
    "_SourceDestination": " for this source-destination pair within the rolling window",
}

_STATIC = {
    "duration_ms": "flow duration in milliseconds",
    "packets": "number of packets",
    "bytes": "number of bytes",
    "min_ps": "minimum packet size",
    "max_ps": "maximum packet size",
    "mean_ps": "mean packet size",
    "stddev_ps": "packet size standard deviation",
    "min_piat_ms": "minimum inter-arrival time (ms)",
    "max_piat_ms": "maximum inter-arrival time (ms)",
    "mean_piat_ms": "mean inter-arrival time (ms)",
    "stddev_piat_ms": "inter-arrival time standard deviation (ms)",
    "syn_packets": "number of SYN packets",
    "cwr_packets": "number of CWR packets",
    "ece_packets": "number of ECE packets",
    "urg_packets": "number of URG packets",
    "ack_packets": "number of ACK packets",
    "psh_packets": "number of PSH packets",
    "rst_packets": "number of RST packets",
    "fin_packets": "number of FIN packets",
}

_DIRECTION = {"src2dst_": "client-to-server ", "dst2src_": "server-to-client ",
              "bidirectional_": "bidirectional "}


def describe_feature(name: str) -> str:
    """Map a column name to the phrasing used in the prompt."""
    for suffix, scope in _SCOPE.items():
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in FEATURE_DESCRIPTIONS:
                return FEATURE_DESCRIPTIONS[base] + scope
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name.startswith("proto_"):
        proto = {"1": "ICMP", "2": "IGMP", "6": "TCP", "17": "UDP", "58": "ICMPv6"}
        return f"flow uses the {proto.get(name.split('_')[1], name)} protocol"
    if name == "Exp_-1":
        return "flow was cut at the 20-packet limit"
    if name == "Exp_0":
        return "flow ended on idle/active timeout"
    for prefix, direction in _DIRECTION.items():
        if name.startswith(prefix):
            tail = name[len(prefix):]
            if tail in _STATIC:
                return direction + _STATIC[tail]
    return name.replace("_", " ")


def build_flow_prompt(explanation: dict, top_n: int = 10) -> str:
    """Q_flow = P_init + P_part2 + P_align (Algorithm 3 lines 8-16)."""
    lines = [P_INIT.format(predicted_class=explanation["predicted_class"]), P_PART2_HEADER]
    for item in explanation["top_flow_features"][:top_n]:
        value = item["value"]
        rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
        lines.append(f"- {describe_feature(item['feature'])} with actual value {rendered}")
    lines.append(P_ALIGN)
    return "\n".join(lines)


def build_payload_prompt(explanation: dict, max_chars: int = 1200) -> str | None:
    """Q_payload = P_payloadPrefix + P_ascii + P_align (Algorithm 3 lines 18-27)."""
    if not explanation.get("payload_specific"):
        return None
    chunks = []
    for item in explanation["top_payloads"]:
        text = item["ascii"].strip(".")
        if not text or set(text) <= {"."}:
            continue
        chunks.append(f"[packet {item['packet_index']}] {text}")
    if not chunks:
        return None
    ascii_blob = "\n".join(chunks)[:max_chars]
    return f"{P_PAYLOAD_PREFIX}\n{ascii_blob}\n{P_ALIGN}"


def build_prompts(explanation: dict, top_n: int = 10) -> dict[str, str]:
    prompts = {"flow": build_flow_prompt(explanation, top_n)}
    payload = build_payload_prompt(explanation)
    if payload:
        prompts["payload"] = payload
    return prompts


class LLMExplainer:
    """Optional zero-shot generation, defaulting to the paper's Llama 3-8B."""

    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 max_new_tokens: int = 400, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device_map)
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def answer(self, prompts: dict[str, str]) -> dict[str, str]:
        """G_exp = R_flow + R_payload (Algorithm 3 line 29)."""
        return {key: self._generate(prompt) for key, prompt in prompts.items()}

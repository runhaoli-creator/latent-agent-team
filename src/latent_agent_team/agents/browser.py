"""
browser.py — Browser/Interactor agent: low-level web + environment actions.

Handles:
  - Mind2Web: DOM element selection (click, type, select, submit)
  - WebShop: Product browsing, search, add-to-cart, buy
  - AgentBench: OS/DB/web tool interactions
  - Selenium/Playwright integration for real browser sessions

Output actions are structured JSON following the role prompt format.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule

logger = logging.getLogger(__name__)


@dataclass
class BrowserState:
    url: str = ""
    dom_snapshot: str = ""
    viewport_text: str = ""
    action_history: List[Dict] = None
    error_count: int = 0

    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []


# ── Action space ──────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    # Mind2Web / Web navigation
    "click": ["element"],
    "type": ["element", "value"],
    "select": ["element", "value"],
    "scroll": ["direction"],
    "navigate": ["url"],
    "submit": [],
    "extract": ["element"],
    # WebShop
    "search": ["query"],
    "add_to_cart": ["product_id"],
    "buy": [],
    # AgentBench / general
    "execute": ["command"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "sql_query": ["query"],
    "api_call": ["endpoint", "method", "body"],
    # Meta
    "done": ["result"],
    "fail": ["reason"],
}


def parse_browser_action(text: str) -> Dict[str, Any]:
    """Extract structured action from model output."""
    # Try JSON first — look for any JSON object in the text
    try:
        # Find all potential JSON objects
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i+1]
                    try:
                        action = json.loads(candidate)
                        if isinstance(action, dict) and ("action" in action or "operation" in action):
                            # Normalize 'operation' key to 'action'
                            if "operation" in action and "action" not in action:
                                action["action"] = action.pop("operation")
                            act = action["action"].lower()
                            if act in VALID_ACTIONS or act in ("search", "type", "navigate", "click", "scroll", "buy", "select"):
                                return action
                    except json.JSONDecodeError:
                        pass
                    start = -1
    except Exception:
        pass

    # Try WebShop-style action formats: search[query], click[element]
    webshop_patterns = [
        (r"search\[([^\]]+)\]", "search", "query"),
        (r"click\[([^\]]+)\]", "click", "element"),
    ]
    for pattern, action_name, key in webshop_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return {"action": action_name, key: m.group(1).strip(), "reasoning": text[:100]}

    # Try regex patterns for common action formats
    patterns = [
        (r"search\s+(?:for\s+)?[\"']([^\"']+)[\"']", "search", "query"),
        (r"click\s+(?:on\s+)?[\"']([^\"']+)[\"']", "click", "element"),
        (r"type\s+[\"']([^\"']+)[\"']\s+(?:in|into)\s+[\"']?([^\"'\n]+)[\"']?", "type", "element"),
        (r"navigate\s+(?:to\s+)?[\"']?(\S+)[\"']?", "navigate", "url"),
    ]

    for pattern, action_name, key in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return {"action": action_name, key: m.group(1).strip(), "reasoning": text[:100]}

    # Last resort: check for common action keywords
    text_lower = text.lower()
    if "buy now" in text_lower or "add to cart" in text_lower or "purchase" in text_lower:
        return {"action": "click", "element": "Buy Now", "reasoning": text[:100]}
    if "search" in text_lower:
        # Extract everything after "search" as the query
        m = re.search(r"search\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            return {"action": "click", "element": m.group(1).strip()[:100], "reasoning": text[:100]}

    # Try to extract candidate number pattern: "element: 3" or just a number
    num_match = re.search(r'(?:element[:\s]*)(\d+)', text, re.IGNORECASE)
    if num_match:
        return {"action": "click", "element": num_match.group(1), "reasoning": text[:100]}
    
    # Try to extract operation type from free text
    detected_op = "click"  # default
    for op in ["select", "type", "scroll", "navigate"]:
        if op.lower() in text.lower():
            detected_op = op.lower()
            break
    
    # Extract any quoted or meaningful text as the element
    m = re.search(r'["\']([^"\']+)["\']', text)
    elem = m.group(1) if m else text.strip()[:80]
    return {"action": detected_op, "element": elem, "reasoning": text[:100]}


class BrowserAgent:
    """
    Browser/Interactor agent: executes low-level actions on web/tool environments.
    """

    ROLE_NAME = "browser"

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        agent_idx: int = 2,
        max_dom_length: int = 1024,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        env_type: str = "mind2web",  # "mind2web" | "webshop" | "agentbench"
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.role_adapter = role_manager.get_adapter(self.ROLE_NAME)
        self.comm_module = comm_module
        self.agent_idx = agent_idx
        self.max_dom_length = max_dom_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.env_type = env_type
        self.state = BrowserState()

    def reset(self, url: str = "", dom: str = "") -> None:
        self.state = BrowserState(url=url, dom_snapshot=dom)

    def _truncate_dom(self, dom: str) -> str:
        """Truncate DOM to fit context window."""
        if len(dom) <= self.max_dom_length:
            return dom
        # Keep first and last portions (headers + action area)
        half = self.max_dom_length // 2
        return dom[:half] + "\n...[TRUNCATED]...\n" + dom[-half:]

    def _build_prompt(
        self,
        observation: str,
        sub_goal: str,
        dom: Optional[str] = None,
        retrieved_evidence: Optional[str] = None,
        incoming_messages: Optional[List[Dict]] = None,
    ) -> str:
        role_prompt = self.role_adapter.get_role_prompt()
        parts = [f"[SYSTEM]\n{role_prompt}\n"]

        if sub_goal:
            parts.append(f"[SUB-GOAL]\n{sub_goal}\n")

        if dom:
            parts.append(f"[DOM/ENVIRONMENT STATE]\n{self._truncate_dom(dom)}\n")
        elif observation:
            parts.append(f"[OBSERVATION]\n{observation[:3000]}\n")

        if retrieved_evidence:
            parts.append(f"[RETRIEVED EVIDENCE]\n{retrieved_evidence[:400]}\n")

        if self.state.action_history:
            recent = self.state.action_history[-5:]
            hist_text = "\n".join(f"  {a.get('action', '?')}: {str(a)[:80]}" for a in recent)
            parts.append(f"[RECENT ACTIONS]\n{hist_text}\n")

        if incoming_messages:
            msg_text = "\n".join(
                f"  [{m.get('sender', '?')}]: {m.get('content', '')[:200]}"
                for m in incoming_messages
            )
            parts.append(f"[TEAM MESSAGES]\n{msg_text}\n")

        parts.append("[ACTION]")
        return "\n".join(parts)

    @torch.no_grad()
    def step(
        self,
        observation: str,
        sub_goal: str = "",
        dom: Optional[str] = None,
        retrieved_evidence: Optional[str] = None,
        incoming_latent: Optional[torch.Tensor] = None,
        incoming_messages: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Execute one browser step.

        Returns:
            action: Structured action dict
            hidden_states: For outgoing latent messages
        """
        prompt = self._build_prompt(observation, sub_goal, dom, retrieved_evidence, incoming_messages)

        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.backbone.cfg.max_seq_len - self.max_new_tokens,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        self.backbone.set_active_adapter(self.ROLE_NAME)

        if incoming_latent is not None and self.comm_module.mode != "text":
            embeds = self.backbone.model.get_input_embeddings()(inputs["input_ids"])
            combined_embeds, combined_mask, _ = self.comm_module(
                hidden_summary=incoming_latent, obs_embeds=embeds
            )
            gen_outputs = self.backbone.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        else:
            gen_outputs = self.backbone.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        new_ids = gen_outputs.sequences[:, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)

        # Extract hidden states
        if hasattr(gen_outputs, "hidden_states") and gen_outputs.hidden_states:
            hidden_states = gen_outputs.hidden_states[-1][-1]
        else:
            fwd_out = self.backbone.model(**inputs, output_hidden_states=True)
            hidden_states = fwd_out.hidden_states[-1]

        action = parse_browser_action(generated_text)
        self.state.action_history.append(action)

        if action.get("action") == "fail":
            self.state.error_count += 1

        return action, hidden_states

    def generate_latent_message(
        self,
        hidden_states: torch.Tensor,
        k: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.role_adapter.summarize_hidden_states(hidden_states, attention_mask, k=k)

    # ── Mind2Web-specific helpers ─────────────────────────────────────────────

    def score_element(
        self,
        element_candidates: List[str],
        sub_goal: str,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Score DOM element candidates for a sub-goal using the backbone.
        Returns index of best candidate.
        Used for Mind2Web element accuracy evaluation.
        """
        tokenizer = self.backbone.tokenizer
        best_idx, best_score = 0, -float("inf")

        self.backbone.set_active_adapter(self.ROLE_NAME)

        for i, elem in enumerate(element_candidates[:20]):  # limit candidates
            text = f"Sub-goal: {sub_goal}\nElement: {elem}\nIs this the correct element? Answer:"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.backbone.model(**inputs)
            # Use log-prob of "Yes" token as score
            yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            score = float(out.logits[0, -1, yes_id].item())
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    # ── WebShop-specific helpers ──────────────────────────────────────────────

    def format_webshop_action(
        self,
        action: Dict[str, Any],
    ) -> str:
        """Convert structured action to WebShop action string."""
        atype = action.get("action", "")
        if atype == "search":
            return f"search[{action.get('query', '')}]"
        elif atype == "click":
            elem = action.get("element", "")
            return f"click[{elem}]"
        elif atype == "add_to_cart":
            return f"click[Add to Cart]"
        elif atype == "buy":
            return f"click[Buy Now]"
        elif atype == "done":
            return f"click[Back to Search]"
        return str(action)

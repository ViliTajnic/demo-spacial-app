from __future__ import annotations

import os
from typing import Optional

import requests

from oci_genai import configured as oci_configured


def _generate_text(prompt: str, config: Optional[dict] = None) -> str:
    cfg = config or {}
    provider = (cfg.get("llm_provider") or os.getenv("LLM_PROVIDER", "ollama")).lower()

    if provider == "oci":
        if not oci_configured(cfg):
            return (
                "OCI GenAI is not configured. Set OCI env vars to enable grounded responses. "
                "Fallback summary: unable to run model triage."
            )

        try:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import ChatDetails, CohereChatRequest, OnDemandServingMode

            oci_config = oci.config.from_file(
                cfg.get("oci_config_file") or os.getenv("OCI_CONFIG_FILE"),
                cfg.get("oci_config_profile") or os.getenv("OCI_CONFIG_PROFILE"),
            )
            client = GenerativeAiInferenceClient(config=oci_config)
            request = ChatDetails(
                compartment_id=cfg.get("oci_compartment_ocid") or os.getenv("OCI_COMPARTMENT_OCID"),
                serving_mode=OnDemandServingMode(model_id=cfg.get("oci_model_id") or os.getenv("OCI_GENAI_MODEL_ID")),
                chat_request=CohereChatRequest(message=prompt, max_tokens=500, temperature=0.2),
            )
            response = client.chat(request)
            text = response.data.chat_response.text
            return text or "No response returned by OCI GenAI model."
        except Exception as exc:  # noqa: BLE001
            return f"OCI GenAI call failed. Error: {exc}"

    ollama_url = cfg.get("ollama_url") or os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = cfg.get("ollama_model") or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("response", "No response text from Ollama.")
    except Exception as exc:  # noqa: BLE001
        return (
            "Local Ollama call failed. Ensure Ollama is running and model is pulled. "
            f"Error: {exc}"
        )


def explain_alert(context_text: str, config: Optional[dict] = None) -> str:
    prompt = (
        "You are an officer assistant for EU location monitoring. "
        "Explain why the alert likely happened and propose next checks. "
        "Use only facts from context and keep it concise.\n\n"
        f"Context:\n{context_text}"
    )
    return _generate_text(prompt, config=config)


def triage_incidents(context_text: str, config: Optional[dict] = None) -> str:
    prompt = (
        "You are a real-time operator watch assistant for EU location monitoring. "
        "Review the incidents and identify which anomalies need immediate operator attention. "
        "Return concise plain text with: 1) top priority incidents, 2) why they matter, 3) next operator action. "
        "Use only facts from context. If nothing is urgent, say that clearly.\n\n"
        f"Context:\n{context_text}"
    )
    return _generate_text(prompt, config=config)


def test_llm(config: Optional[dict] = None) -> tuple[bool, str]:
    text = explain_alert('{"healthcheck":"ok"}', config=config)
    lowered = text.lower()
    if "failed" in lowered and "error:" in lowered:
        return False, text
    return True, "LLM reachable."

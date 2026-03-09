from __future__ import annotations

import os
from typing import Optional


def configured(config: Optional[dict] = None) -> bool:
    cfg = config or {}
    required = [
        cfg.get("oci_model_id") or os.getenv("OCI_GENAI_MODEL_ID"),
        cfg.get("oci_compartment_ocid") or os.getenv("OCI_COMPARTMENT_OCID"),
        cfg.get("oci_config_file") or os.getenv("OCI_CONFIG_FILE"),
        cfg.get("oci_config_profile") or os.getenv("OCI_CONFIG_PROFILE"),
    ]
    return all(required)


def explain_alert(context_text: str, config: Optional[dict] = None) -> str:
    cfg = config or {}
    if not configured(cfg):
        return (
            "OCI GenAI is not configured. Set OCI env vars to enable grounded explanations. "
            "Fallback summary: alert likely triggered by a rule or device anomaly in the recent timeline."
        )

    import oci
    from oci.generative_ai_inference import GenerativeAiInferenceClient
    from oci.generative_ai_inference.models import ChatDetails, CohereChatRequest, OnDemandServingMode

    oci_config = oci.config.from_file(
        cfg.get("oci_config_file") or os.getenv("OCI_CONFIG_FILE"),
        cfg.get("oci_config_profile") or os.getenv("OCI_CONFIG_PROFILE"),
    )

    client = GenerativeAiInferenceClient(config=oci_config)
    prompt = (
        "You are an officer assistant. Explain this alert in concise operational language. "
        "Use only facts from context and include a confidence statement.\n\n"
        f"Context:\n{context_text}"
    )

    request = ChatDetails(
        compartment_id=cfg.get("oci_compartment_ocid") or os.getenv("OCI_COMPARTMENT_OCID"),
        serving_mode=OnDemandServingMode(model_id=cfg.get("oci_model_id") or os.getenv("OCI_GENAI_MODEL_ID")),
        chat_request=CohereChatRequest(message=prompt, max_tokens=350, temperature=0.2),
    )

    response = client.chat(request)
    text = response.data.chat_response.text
    return text or "No response returned by OCI GenAI model."

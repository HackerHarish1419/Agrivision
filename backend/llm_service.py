"""
AgriVisionAI — LLM Service (Groq Integration)
Generates context-aware recovery recommendations using Groq API.
All config from environment variables — nothing hardcoded.
"""

import os
import json
import re
import httpx
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()


class LLMService:
    """Context-aware LLM layer using Groq API."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = None
        self.configured = False

        if self.api_key and self.api_key != "your_groq_api_key_here":
            self.configured = True
            print(f"✅ Groq API configured (model: {self.model})")
        else:
            print("⚠️  API_KEY not set in .env — LLM recommendations will be unavailable")

    def _build_prompt(self, crop: str, disease: str, confidence: float) -> str:
        """Build structured prompt for the LLM."""
        return f"""You are an expert agricultural pathologist. Your goal is to give fast, undeniable, highly-specific advice. 
DO NOT write long paragraphs. Your answers must be incredibly concise, actionable, and punchy.

A farmer submitted a {crop} leaf image. The AI Vision core detected:
- **Detected Disease:** {disease}
- **Confidence:** {confidence:.1%}

Provide a fast recovery plan. Your response MUST be valid JSON with EXACTLY these fields:

{{
  "disease_summary": "A punchy 1-2 sentence maximum summary of what this disease is and its root cause.",
  "recovery_steps": ["Step 1: [Action in <10 words]", "Step 2: [Action in <10 words]", "Step 3: [Action in <10 words]"],
  "organic_treatment": ["Specific organic fungicide name & dosage", "Alternative organic spray"],
  "chemical_treatment": ["EXACT chemical name (e.g., Mancozeb 75% WP) + dosage", "Alternative chemical + frequency"],
  "time_to_recovery": "E.g., '2-3 weeks'",
  "preventive_measures": ["Short preventive measure 1", "Short preventive measure 2"],
  "severity": "low OR medium OR high"
}}

CRITICAL RULES:
- Be hyper-specific to {disease} in {crop}.
- Keep every single string/array item under 15 words. DO NOT babble.
- Name ACTUAL chemical compounds and ACTUAL dosages. Do not say "consult a local store".
- Respond ONLY with the raw JSON object, no markdown codeblocks, no text before or after."""

    def _parse_response(self, text: str) -> Dict:
        """Parse LLM response, handling potential formatting issues."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return raw text as summary if all parsing fails
        return {
            "disease_summary": text[:500],
            "recovery_steps": ["Please consult an agricultural expert for detailed steps."],
            "organic_treatment": ["Consult local organic farming resources."],
            "chemical_treatment": ["Consult a certified agronomist for chemical recommendations."],
            "time_to_recovery": "Varies — seek professional assessment",
            "preventive_measures": ["Regular crop monitoring", "Maintain field hygiene"],
            "severity": "medium"
        }

    async def get_recommendation(self, crop: str, disease: str, confidence: float) -> Dict:
        """Get disease recovery recommendation from LLM."""
        if not self.configured:
            return {
                "success": False,
                "error": "LLM service not configured. Set GROQ_API_KEY in .env file.",
                "recommendation": None
            }

        # Handle healthy plants
        if "healthy" in disease.lower():
            return {
                "success": True,
                "recommendation": {
                    "disease_summary": f"The {crop} leaf appears healthy with no visible signs of disease. Continue current maintenance practices.",
                    "recovery_steps": ["No treatment needed — the plant appears healthy."],
                    "organic_treatment": ["Continue regular composting and mulching schedules."],
                    "chemical_treatment": ["No chemical treatment required for healthy plants."],
                    "time_to_recovery": "N/A — plant is healthy",
                    "preventive_measures": [
                        "Maintain regular watering schedule",
                        "Monitor weekly for early signs of disease",
                        "Ensure proper spacing for air circulation",
                        "Apply preventive organic fungicide if in a disease-prone area"
                    ],
                    "severity": "low"
                }
            }

        try:
            prompt = self._build_prompt(crop, disease, confidence)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert agricultural pathologist. Always respond with valid JSON only."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1500,
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    raise Exception(f"Groq API Error {response.status_code}: {response.text}")

                data = response.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                recommendation = self._parse_response(response_text)

            return {
                "success": True,
                "recommendation": recommendation
            }

        except Exception as e:
            print(f"❌ GROQ API ERROR: {type(e).__name__} - {str(e)}")
            return {
                "success": False,
                "error": f"LLM request failed: {str(e)}",
                "recommendation": None
            }

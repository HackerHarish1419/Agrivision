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
        return f"""You are an advanced agricultural AI system analyzing farm telemetry. 

A farmer submitted a {crop} leaf image. The vision model detected:
- **Detected Disease:** {disease}
- **Confidence:** {confidence:.1%}

Generate a hardcore, concise, hackathon-style tactical readout.
Return ONLY valid JSON with EXACTLY these fields:

{{
  "severity": "Moderate OR Severe OR Low",
  "location_context": "Mangalore, Karnataka, India",
  "time_context": "Early Morning - ideal spraying window",
  "recommendations": [
    "1. Actionable step one",
    "2. Actionable step two",
    "3. Actionable step three",
    "4. Actionable step four",
    "5. Actionable step five"
  ],
  "recovery_time": "14-21 days with immediate intervention",
  "preventive_note": "Pre-monsoon conditions in Kerala increase fungal risk - inspect weekly"
}}

CRITICAL RULES:
- The JSON keys must match exactly.
- Make the recommendations highly technical, concise, and specific to {disease}.
- Do not use markdown, return only the raw JSON string."""

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
            "severity": "Unknown",
            "location_context": "Unknown Location",
            "time_context": "Unknown Time",
            "recommendations": ["System error parsing LLM output.", text[:200]],
            "recovery_time": "Unknown",
            "preventive_note": "Please consult a specialist."
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
                    "severity": "None",
                    "location_context": "Mangalore, Karnataka, India",
                    "time_context": "Current Time - Standard ops",
                    "recommendations": [
                        "1. No immediate action required; plant is healthy",
                        "2. Continue standard preventive fertilization schedule",
                        "3. Maintain regular irrigation monitoring"
                    ],
                    "recovery_time": "N/A - Healthy",
                    "preventive_note": "Monitor bi-weekly as seasonal shifts approach"
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

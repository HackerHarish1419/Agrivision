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
        return f"""You are an expert agricultural pathologist and crop disease specialist.

A farmer has submitted a leaf image from their field. The AI vision model has analyzed the image and detected the following:

- **Crop:** {crop}
- **Detected Disease:** {disease}
- **Detection Confidence:** {confidence:.1%}

Based on this diagnosis, provide a comprehensive recovery and treatment plan. Your response MUST be valid JSON with exactly these fields:

{{
  "disease_summary": "A 2-3 sentence summary explaining what this disease is, how it affects the crop, and its typical cause (pathogen, environmental conditions, etc.)",
  "recovery_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "organic_treatment": ["Organic option 1 with application details", "Organic option 2 with application details"],
  "chemical_treatment": ["Chemical 1 with dosage and frequency", "Chemical 2 with dosage and frequency"],
  "time_to_recovery": "Estimated recovery timeline (e.g., '2-4 weeks with proper treatment')",
  "preventive_measures": ["Preventive measure 1", "Preventive measure 2", "Preventive measure 3"],
  "severity": "low OR medium OR high"
}}

IMPORTANT:
- Be specific to {crop} and {disease} — do NOT give generic advice
- Include actual chemical names, dosages, and application frequencies
- Recovery steps should be in chronological order of priority
- Severity should reflect how damaging this disease is to yield if untreated
- Respond with ONLY the JSON object, no markdown, no explanation outside the JSON"""

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

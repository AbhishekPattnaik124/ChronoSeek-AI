import os
import json
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class QueryDecomposer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.is_active = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.is_active = True
            except Exception as e:
                print(f"LLM Initialization Warning: {e}")

    def decompose(self, query: str) -> Dict:
        """
        Decomposes a complex query into sub-queries and filters using an LLM.
        """
        if not self.is_active:
            return self._fallback_decomposition(query)

        prompt = f"""
        Analyze this video search query: "{query}"
        Decompose it into visual sub-queries for an embedding search and extract filters.
        
        Return ONLY a JSON object:
        {{
          "sub_queries": ["list of purely visual components"],
          "temporal_filter": {{"start": "HH:MM:SS" or null, "end": "HH:MM:SS" or null}},
          "spatial_hints": ["list of spatial relationships like 'near', 'inside'"],
          "query_type": "temporal" | "spatial" | "temporal+spatial" | "simple"
        }}
        
        Example: "person near entrance after 6PM" 
        -> {{"sub_queries": ["person", "entrance"], "temporal_filter": {{"start": "18:00:00", "end": null}}, "spatial_hints": ["near entrance"], "query_type": "temporal+spatial"}}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Cleanup potential markdown formatting
            text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            print(f"LLM Decomposition Error: {e}")
            return self._fallback_decomposition(query)

    def _fallback_decomposition(self, query: str) -> Dict:
        """Simple rule-based decomposition as a fallback."""
        return {
            "sub_queries": [query],
            "temporal_filter": {"start": None, "end": None},
            "spatial_hints": [],
            "query_type": "simple"
        }

if __name__ == "__main__":
    # Test logic
    decomposer = QueryDecomposer()
    test_queries = [
        "person near entrance carrying a bag after 6PM",
        "red car parked next to a blue truck before noon",
        "two people talking in the corridor"
    ]
    
    print("--- Decomposition Test ---")
    for q in test_queries:
        result = decomposer.decompose(q)
        print(f"Query: {q}")
        print(json.dumps(result, indent=2))
        print("-" * 20)

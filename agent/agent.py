"""
Final layer of the project. Just calls upon the agent to run our entire architecture. 
Done at last omg. If you are reading this you a real one gng
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "V2" / "source"))
import models
from pipeline import run_battery_pipeline

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

MODEL_NAME = "gemini-2.5-flash"  # swap to another Gemini model name if you prefer

SYSTEM_PROMPT = """You are a battery health assistant. You have just been given the
complete output of a 5-component ML pipeline for one battery's current cycle:
  - predicted_soh: state of health (0-1, fraction of rated capacity remaining)
  - predicted_rul_cycles: estimated cycles remaining until end-of-life
  - needs_replacement + replacement_confidence: binary recommendation + model confidence.
  - degradation_stage: Healthy / Transitional / Critical (from unsupervised clustering,
    NOT the same signal as needs_replacement and it can disagree, and that's meaningful)
  - is_anomalous + anomaly_top_reasons: whether this cycle's readings look statistically
    unusual compared to the training fleet, and which features are most responsible

Answer the user's question using ONLY this data. Rules:
  - If is_anomalous is true, say so explicitly and caveat that the other predictions
    (soh/rul/replacement) may be less trustworthy for this specific reading and don't
    just report the anomaly flag as one more data point among equals.
  - If degradation_stage and needs_replacement seem to disagree (e.g. stage is
    Transitional but needs_replacement is false), explain why that's not a
    contradiction as they answer different questions.
  - Replacement confidence is confidence with which a battery is to be replaced. So invert it for a healthy battery
  - Be direct and concrete. State the actual numbers. Don't hedge with vague
    language when the data gives you something specific to say.
  - This model was trained on ~140 batteries with known validation limitations
    (SOH R2=0.90, RUL R2=0.62, replacement F1=0.78/recall=0.93/precision=0.68).
  - If the user asks how much to trust a prediction give the real number.
  - Answer everything as concise and professional as possible and mention the data point wise.
  - Mention Executed after finishing the answer.
"""


def ask_battery_agent(battery_id: str, question: str, df=None) -> str:
    if genai is None:
        raise ImportError("pip install google-genai first")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable first")

    if df is None:
        df = models.load_clean_data()

    pipeline_result = run_battery_pipeline(battery_id, df)

    client = genai.Client()  # picks up GEMINI_API_KEY/GOOGLE_API_KEY automatically
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Pipeline output for battery {battery_id}:\n"
                 f"{json.dumps(pipeline_result, indent=2)}\n\n"
                 f"User question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2000,
        ),
    )
    return response.text


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python3 agent_gemini.py <battery_id> "<question>"')
        sys.exit(1)

    battery_id, question = sys.argv[1], sys.argv[2]
    print(f"\n--- Raw pipeline output for {battery_id} ---")
    df = models.load_clean_data()
    result = run_battery_pipeline(battery_id, df)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print(f"\n--- Agent's answer ---")
    print(ask_battery_agent(battery_id, question, df))
#!/usr/bin/env python3
"""
Verbalized Assumptions script
uses local models for models < 70B, otherwise API

Usage:
    python get_assumptions.py input.csv output.csv model prompt_type [--api-key KEY] [--sample-n N] [--max-user-turns N] [--max-workers N]

Examples:
   python3 get_assumptions.py \
    AITA-YTA_prompts.csv \
    AITA-YTA_gemini_4dims.csv \
    gemini \
    4dims # this is the dimensions of user rightness, objectivity seeking, validation seeking, and user information advantage

    python3 get_assumptions.py \
    resps.csv \
    resps_gemini_supporttypes.csv \
    gemini \
    supporttypes \ # these are the five dimensions of support from the taxonomy
    --max-user-turns 1

"""

import sys
import os
import re
import pandas as pd
import time
import random
import requests
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from openai import AzureOpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== API Configuration ==========

# Azure OpenAI (GPT-4o) Configuration
AZURE_OPENAI_ENDPOINT ="TODO"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT_NAME = 'gpt-4o'

# Vertex AI (Gemini) Configuration
GEMINI_PROJECT_ID = 'TODO'
GEMINI_LOCATION ='TODO'
GEMINI_SERVICE_ACCOUNT_FILE = 'TODO.json'
GEMINI_MODEL_ID = "gemini-2.5-pro" 

# Vertex AI (Llama70b) Configuration
LLAMA_PROJECT_ID = "TODO"
LLAMA_LOCATION = 'TODO'
LLAMA_MODEL_ID = "meta/llama-3.3-70b-instruct-maas"
LOCAL_MODEL_PATHS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    'qwen-14b':'Qwen/Qwen2.5-14B-Instruct',
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    'qwen-72b':'Qwen/Qwen2.5-72B-Instruct',
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    'llama-3b':'meta-llama/Llama-3.2-3B-Instruct'
}

# Global credentials for Vertex AI (will be refreshed as needed)
creds = None
creds_lock = Lock()

# Global variables for local inference models
local_model = None
local_tokenizer = None
local_model_lock = Lock()

def init_vertex_credentials():
    """Initialize Vertex AI credentials."""
    global creds
    if creds is None:
        creds = service_account.Credentials.from_service_account_file(
            GEMINI_SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh(Request())

# ========== Model Configuration ==========
AVAILABLE_MODELS = {
    # API-based models
    "llama-70b": "llama70b",
    "gpt4o": "gpt4o",
    "gemini": "gemini",
    # Local inference models
    "qwen-0.5b": "qwen-0.5b",
    "qwen-1.5b": "qwen-1.5b",
    "qwen-3b": "qwen-3b",
    "qwen-32b": "qwen-32b",
    'qwen-14b':'qwen-14b',
    'qwen-72b':'qwen-72b',
    "qwen-7b": "qwen-7b",
    "llama-8b": "llama-8b",
    'llama-3b':'llama-3b'
    }

# Models that use local inference
LOCAL_INFERENCE_MODELS = [
    "qwen-0.5b",
    "qwen-1.5b",
    "qwen-3b",
    "qwen-7b",
    "qwen-32b",
    "llama-8b",
    'qwen-14b',
    'qwen-72b',
    'llama-3b'
]

AVAILABLE_PROMPTS = {
    "openended": "openended",
    "twostep":"twostep",
    "4dims": "4dims",
    'ten':'ten',
    'base':'base',
    "support": "support",
    'cot':'cot',
    "supporttypes": "supporttypes",
    'supporttypestwostep':'supporttypestwostep'
    }

# ========== Local Inference Functions ==========
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_local_model(model_key: str):
    """Load local inference model and tokenizer in 4-bit."""
    global local_model, local_tokenizer

    if model_key not in LOCAL_MODEL_PATHS:
        raise ValueError(f"Unknown local model: {model_key}")

    MODEL_NAME = LOCAL_MODEL_PATHS[model_key]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading local model (4-bit): {MODEL_NAME}")
    print(f"Using device: {DEVICE}")

    local_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    local_tokenizer.padding_side = "left"

    if DEVICE == "cuda":
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",          # best for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16  # use float16 if no bf16 support
        )

        local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
             device_map={"": 0}
            #device_map="auto",
        )

    else:
        # CPU fallback (no 4-bit)
        local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

    local_model.eval()
    print("✅ Local model loaded successfully (4-bit if CUDA available)!")

def old_load_local_model(model_key: str):
    """Load local inference model and tokenizer."""
    global local_model, local_tokenizer
    
    if model_key not in LOCAL_MODEL_PATHS:
        raise ValueError(f"Unknown local model: {model_key}")
    
    MODEL_NAME = LOCAL_MODEL_PATHS[model_key]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading local model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    
    local_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    local_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    local_model.eval()
    
    print(f"✅ Local model loaded successfully!")

def make_local_inference_call_batch(prompts: list[str], max_tokens: int = 5000) -> list[str]:
    """
    True batched local inference:
    - prompts: list of raw prompt strings (user content)
    - returns: list of generated strings, same length/order
    """
    try:
        input_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            input_texts.append(
                local_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # Ensure pad token exists for batching
        if local_tokenizer.pad_token_id is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token

        # Tokenize as a batch (padding -> real batching)
        inputs = local_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(local_model.device)

        with local_model_lock:
            with torch.no_grad():
                outputs = local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=local_tokenizer.eos_token_id,
                )

        # Decode each sequence, removing its own prompt tokens
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        generations = []
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i][input_lens[i]:]
            generations.append(local_tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return generations

    except Exception as e:
        print(f"[ERROR] Batched local inference failed: {e}")
        return [f"[INFERENCE_ERROR: {e}]" for _ in prompts]


def make_local_inference_call(prompt: str, max_tokens: int = 5000) -> str:
    """Single-prompt wrapper (keeps old call sites working)."""
    return make_local_inference_call_batch([prompt], max_tokens=max_tokens)[0]

# ========== API Call Functions ==========

def make_api_call_llama70b(client, model_name: str, prompt: str, max_retries: int = 5) -> str:
    """Make API call to Llama70b via Vertex AI."""
    init_vertex_credentials()
    
    url = (
        f"https://{LLAMA_LOCATION}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{LLAMA_PROJECT_ID}/locations/{LLAMA_LOCATION}/endpoints/openapi/chat/completions"
    )

    payload = {
        "model": LLAMA_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
    }

    for attempt in range(1, max_retries + 1):
        with creds_lock:
            if not creds.valid or creds.expired:
                creds.refresh(Request())
            token = creds.token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}") from e
    
    return "ERROR"


def make_api_call_gpt4o(client, model_name: str, prompt: str, system_prompt: str = "",
                        max_retries: int = 3, base_delay: float = 1.0) -> str:
    """Make API call to GPT-4o via Azure OpenAI."""
    for attempt in range(1, max_retries + 1):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=5000,  # Consistent with other models
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API call failed (attempt {attempt}/{max_retries}): {e}")

            if attempt == max_retries:
                break

            # Exponential backoff with jitter
            sleep_time = base_delay * (2 ** (attempt - 1))
            sleep_time += random.uniform(0, 0.5)
            time.sleep(sleep_time)

    return "ERROR"


def make_api_call_gemini(client, model_name: str, prompt: str, max_tokens: int = 5000) -> str:
    """Make API call to Gemini via Vertex AI."""
    init_vertex_credentials()
    
    url = (
        f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{GEMINI_PROJECT_ID}/locations/{GEMINI_LOCATION}/publishers/google/models/{GEMINI_MODEL_ID}:generateContent"
    )
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }
    
    try:
        with creds_lock:
            if not creds.valid or creds.expired:
                creds.refresh(Request())
            token = creds.token
            
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return f"[API_ERROR: {e}]"
    
    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except Exception as e:
        print(f"[ERROR] Failed to parse response JSON: {e}")
        return f"[PARSE_ERROR: {e}]"


def make_api_call(client, model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Route API calls to appropriate model (API or local inference)."""
    # Check if it's a local inference model
    if model_name in LOCAL_INFERENCE_MODELS:
        return make_local_inference_call(prompt)
    
    # Otherwise route to API models
    if model_name == "gpt4o":
        return make_api_call_gpt4o(client, model_name, prompt, system_prompt)
    elif model_name == "gemini":
        return make_api_call_gemini(client, model_name, prompt)
    elif model_name == "llama70b":
        return make_api_call_llama70b(client, model_name, prompt)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ========== Helper Functions ==========

def get_sorted_user_cols(df: pd.DataFrame):
    """Return user columns like ['user_1', 'user_2', ...] sorted by numeric suffix."""
    user_cols = [c for c in df.columns if re.match(r"user_\d+$", c)]
    if len(user_cols) < 1:
        # Check for fallback columns
        fallback_candidates = []
        
        if 'prompt' in df.columns:
            fallback_candidates.append('prompt')
        
        if 'user_text' in df.columns:
            fallback_candidates.append('user_text')
        
        if not fallback_candidates:
            # No viable columns found
            return []
        
        # Return first valid fallback
        return fallback_candidates[:1]
    
    # Sort by numeric suffix
    user_cols = sorted(user_cols, key=lambda c: int(c.split("_")[1]))
    return user_cols

def build_prompt_ten(history_str: str, new_user_text: str) -> str:
    """openended prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top ten possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
    ...
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1.

"""
    return prompt

def build_prompt_openended(history_str: str, new_user_text: str) -> str:
    """openended prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top three possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1 that sums to 1 across the three models.

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_twostep(history_str: str, new_user_text: str) -> str:
    """openended prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top three possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1 that sums to 1 across the three models.

Next, provide your reasoning for how to respond given these possible mental models. 

On a new line, write your reasoning under the heading:

REASONING:

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_4dims(history_str: str, new_user_text: str) -> str:
    """4 dimensions prompt: validation seeking, rightness, information advantage, objectivity."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about User A along the following dimensions:

1. Belief about how much User A is seeking validation/support.
2. Belief about how right User A is in the situation.
3. Belief about whether User A has more information than you (the model).
4. Belief about whether User A is seeking an objective perspective.

Then output ONLY a valid JSON object in the following structure:

{{
  "mental_model": {{
    "beliefs": {{
      "validation_seeking": {{
        "score": 0.0,
        "explanation": ""
      }},
      "user_rightness": {{
        "score": 0.0,
        "explanation": ""
      }},
      "user_information_advantage": {{
        "score": 0.0,
        "explanation": ""
      }},
      "objectivity_seeking": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt


def build_prompt_supporttypes(history_str: str, new_user_text: str) -> str:
    """Detailed social support taxonomy prompt."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about the extent to which User A is seeking different types of support:

1. **Emotional Support** - Seeking opportunities for confiding, sympathetic listening, or caring behaviors:
   - Confiding/listening: seeking empathetic understanding or wanting someone to listen to private feelings or problems
   - Affection: seeking expressions of love, care, or emotional closeness
   - Esteem support: seeking validation of self-worth, acceptance despite difficulties
   - Being there: seeking unconditional availability or presence
   - Comforting touch: seeking physical comfort or affection 

2. **Social Contact and Companionship** - Seeking positive social interaction:
   - Companionship: wanting to spend time with others, do activities together
   - Positive interaction: seeking to joke, talk about interests, engage in diversionary activities
   - Shared activities: wanting to do fun things with others

3. **Belonging Support** - Seeking connection to a group or community:
   - Social integration: wanting to feel part of a group with common interests
   - Group inclusion: seeking comfort, security, or identity through group membership
   - Sense of belonging: wanting to feel included and connected

4. **Information and Guidance Support** - Seeking knowledge, advice, or problem-solving help:
   - Advice/guidance: seeking solutions, feedback, or direction
   - Information: seeking facts, explanations, or understanding of situations
   - Cognitive guidance: seeking help in defining or coping with problems

5. **Tangible Support** - Seeking practical or instrumental assistance:
   - Material aid: seeking financial help, resources, or physical objects
   - Practical assistance: seeking help with tasks, chores, or concrete actions
   - Reliable alliance: seeking assurance that others will provide tangible help

Treat these as *probabilistic beliefs* that may co-exist. These dimensions are independent and do NOT need to sum to 1. Each score should be between 0 and 1.

Then output ONLY a valid JSON object in the following structure:
{{
  "mental_model": {{
    "support_seeking": {{
      "emotional_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "social_companionship": {{
        "score": 0.0,
        "explanation": ""
      }},
      "belonging_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "information_guidance": {{
        "score": 0.0,
        "explanation": ""
      }},
      "tangible_support": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_supporttypestwostep(history_str: str, new_user_text: str) -> str:
    """Detailed social support taxonomy prompt."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about the extent to which User A is seeking different types of support:

1. **Emotional Support** - Seeking opportunities for confiding, sympathetic listening, or caring behaviors:
   - Confiding/listening: seeking empathetic understanding or wanting someone to listen to private feelings or problems
   - Affection: seeking expressions of love, care, or emotional closeness
   - Esteem support: seeking validation of self-worth, acceptance despite difficulties
   - Being there: seeking unconditional availability or presence
   - Comforting touch: seeking physical comfort or affection 

2. **Social Contact and Companionship** - Seeking positive social interaction:
   - Companionship: wanting to spend time with others, do activities together
   - Positive interaction: seeking to joke, talk about interests, engage in diversionary activities
   - Shared activities: wanting to do fun things with others

3. **Belonging Support** - Seeking connection to a group or community:
   - Social integration: wanting to feel part of a group with common interests
   - Group inclusion: seeking comfort, security, or identity through group membership
   - Sense of belonging: wanting to feel included and connected

4. **Information and Guidance Support** - Seeking knowledge, advice, or problem-solving help:
   - Advice/guidance: seeking solutions, feedback, or direction
   - Information: seeking facts, explanations, or understanding of situations
   - Cognitive guidance: seeking help in defining or coping with problems

5. **Tangible Support** - Seeking practical or instrumental assistance:
   - Material aid: seeking financial help, resources, or physical objects
   - Practical assistance: seeking help with tasks, chores, or concrete actions
   - Reliable alliance: seeking assurance that others will provide tangible help

Treat these as *probabilistic beliefs* that may co-exist. These dimensions are independent and do NOT need to sum to 1. Each score should be between 0 and 1.

Then output ONLY a valid JSON object in the following structure:
{{
  "mental_model": {{
    "support_seeking": {{
      "emotional_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "social_companionship": {{
        "score": 0.0,
        "explanation": ""
      }},
      "belonging_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "information_guidance": {{
        "score": 0.0,
        "explanation": ""
      }},
      "tangible_support": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}


Next, provide your reasoning for how to respond given these possible mental models. 

On a new line, write your reasoning under the heading:

REASONING:

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_with_history(history_str: str, new_user_text: str, prompt_type: str) -> str:
    """Build prompt based on specified type."""
    if prompt_type == "openended":
        return build_prompt_openended(history_str, new_user_text)
    elif prompt_type == 'twostep':
        return build_prompt_twostep(history_str, new_user_text)
    elif prompt_type == 'ten':
        return build_prompt_ten(history_str, new_user_text)
    elif prompt_type == "4dims":
        return build_prompt_4dims(history_str, new_user_text)
    elif prompt_type == "supporttypes":
        return build_prompt_supporttypes(history_str, new_user_text)
    elif prompt_type == 'supporttypestwostep':
        return build_prompt_supporttypestwostep(history_str, new_user_text)
    elif prompt_type == 'cot':
        return new_user_text+' Think step by step.'
    elif prompt_type =='base':
        return new_user_text
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def run_for_row(client, model_name: str, row: pd.Series, user_cols: list[str], 
                prompt_type: str, max_user_turns: int | None = None):
    """
    For a single conversation row:
      - Iterate over user_1, user_2, ...
      - At each user turn, call model to produce mental model + reply.
      - Accumulate history so later turns condition on earlier ones.
    """
    history_str = ""  # plain text "User: ..., AI: ..." history
    
    if 'history' in row.index:
        history_str = row['history']

    outputs = []

    count = 0
    for u_col in user_cols:
        if max_user_turns is not None and count >= max_user_turns:
            break

        if u_col not in row.index:
            continue

        user_text = row[u_col]
        if not isinstance(user_text, str) or not user_text.strip():
            continue

        user_text = user_text.strip()

        # Build the prompt with accumulated history
        prompt = build_prompt_with_history(history_str, user_text, prompt_type)
        
        # Call the model
        model_output = make_api_call(client, model_name, prompt)

        # Record this turn
        turn_idx = count
        outputs.append({
            "user_turn_index": turn_idx,
            "user_col": u_col,
            "user_text": user_text,
            "model_output": model_output,
        })

        # Update history for next turn
        if history_str:
            history_str += "\n"
        history_str += f"User: {user_text}\nAI: {model_output}"

        count += 1

    return outputs


def process_row_wrapper(args):
    """Wrapper function for parallel processing."""
    conv_id, row, client, model_name, user_cols, prompt_type, max_user_turns = args
    try:
        per_row_outputs = run_for_row(
            client, model_name, row, 
            user_cols=user_cols, 
            prompt_type=prompt_type, 
            max_user_turns=max_user_turns
        )
        records = []
        
        # Get all non-user columns as metadata
        metadata = {}
        for col in row.index:
            if not re.match(r"user_\d+$", col):
                metadata[col] = row[col]
        
        for o in per_row_outputs:
            rec = {
                "conv_id": conv_id,
                **metadata,
                **o,
            }
            records.append(rec)
        return conv_id, records, None
    except Exception as e:
        print(f"[ERROR] Failed processing conv_id={conv_id}: {e}")
        return conv_id, [], str(e)


def load_existing_results(output_csv: str):
    """Load existing results and return processed conv_ids."""
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'conv_id' in existing_df.columns:
                processed_conv_ids = set(existing_df['conv_id'].unique())
                print(f"Found existing output file with {len(existing_df)} rows")
                print(f"Already processed {len(processed_conv_ids)} conversations")
                return existing_df.to_dict('records'), processed_conv_ids
            else:
                print(f"Warning: Existing file found but no 'conv_id' column. Starting fresh.")
                return [], set()
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}. Starting fresh.")
            return [], set()
    return [], set()


def initialize_client(model_key: str, api_key: str | None = None):
    """Initialize appropriate client based on model."""
    # Check if it's a local inference model
    if model_key in LOCAL_INFERENCE_MODELS:
        load_local_model(model_key)
        return None  # No client needed for local models
    
    # API-based models
    if model_key == "gpt4o":
        # Read API key from file if not provided
        if api_key is None:
            try:
                with open('keys.txt', 'r') as f:
                    api_key = [line.rstrip('\n') for line in f][0]
            except:
                raise ValueError("Could not read Azure OpenAI API key from keys.txt")
        
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=api_key,
        )
        return client
    elif model_key == "gemini":
        # Gemini uses service account, no client needed
        init_vertex_credentials()
        return None
    elif model_key == "llama-70b":
        # Llama70b uses service account, no client needed
        init_vertex_credentials()
        return None
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ========== Main Function ==========

def main(input_csv: str, output_csv: str, model_key: str, prompt_type: str,
         api_key: str | None = None, sample_n: int | None = None, 
         first_n: int | None = None, max_user_turns: int | None = None, max_workers: int = 4,batch_size: int = 24):
    """
    Main function for processing with multiple model APIs and local inference.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model_key: Model to use (llama-70b, gpt4o, gemini, qwen-32b, qwen-7b, llama-8b)
        prompt_type: Prompt type (openended, 4dims, support, supporttypes)
        api_key: API key (for GPT-4o, optional if in keys.txt)
        sample_n: Number of rows to sample (None for all)
        max_user_turns: Maximum user turns per conversation (None for all)
        max_workers: Number of parallel workers (default: 4)
    """
    # Load existing results if output file exists
    all_records, processed_conv_ids = load_existing_results(output_csv)
    
    # Validate inputs
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    if prompt_type not in AVAILABLE_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(AVAILABLE_PROMPTS.keys())}")
    
    # Initialize client or load model
    client = initialize_client(model_key, api_key)
    model_name = AVAILABLE_MODELS[model_key]
    
    # Load data
    df = pd.read_csv(input_csv)
    user_cols = get_sorted_user_cols(df)
    if not user_cols:
        raise ValueError("No columns matching 'user_<n>' found in input CSV.")
    
    if sample_n is not None and sample_n < len(df):
        df_sub = df.sample(sample_n,random_state=42).copy()
    elif first_n is not None:
        df_sub = df.head(first_n).copy()
    else:
        df_sub = df.copy()

    # Filter out already processed conversations
    df_remaining = df_sub[~df_sub.index.isin(processed_conv_ids)].copy()
    
    if len(df_remaining) == 0:
        print("✅ All conversations already processed!")
        return

    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Model: {model_key} ({'local inference' if model_key in LOCAL_INFERENCE_MODELS else 'API'})")
    print(f"Prompt type: {prompt_type}")
    print(f"Total conversations: {len(df_sub)}")
    print(f"Already completed: {len(processed_conv_ids)}")
    print(f"Remaining to process: {len(df_remaining)}")
    print(f"Max user turns: {max_user_turns if max_user_turns else 'All'}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*80}\n")
    
    completed_count = len(processed_conv_ids)
    save_lock = Lock()
    
    # --------------------------
    # TRUE BATCHED LOCAL INFERENCE
    # --------------------------
    if model_key in LOCAL_INFERENCE_MODELS:

        def row_metadata(row: pd.Series) -> dict:
            md = {}
            for col in row.index:
                if not re.match(r"user_\d+$", col):
                    md[col] = row[col]
            return md

        df_rem = df_remaining.copy()

        # per-conversation running history
        histories: dict[int, str] = {}
        for conv_id, row in df_rem.iterrows():
            if "history" in row.index and isinstance(row["history"], str):
                histories[conv_id] = row["history"]
            else:
                histories[conv_id] = ""

        conv_ids = list(df_rem.index)

        # limit turns if requested
        turn_cols = user_cols
        if max_user_turns is not None:
            turn_cols = turn_cols[:max_user_turns]

        for start in range(0, len(conv_ids), batch_size):
            batch_conv_ids = conv_ids[start : start + batch_size]
            batch_rows = [df_rem.loc[cid] for cid in batch_conv_ids]
            batch_metadata = [row_metadata(r) for r in batch_rows]

            # active convs in this batch
            active = [True] * len(batch_conv_ids)

            # iterate user turns; batch across conversations at each turn
            for u_col in turn_cols:
                prompts = []
                idx_map = []  # maps prompt idx -> batch index
                user_texts = []

                for bi, (cid, row) in enumerate(zip(batch_conv_ids, batch_rows)):
                    if not active[bi]:
                        continue
                    if u_col not in row.index:
                        active[bi] = False
                        continue

                    ut = row[u_col]
                    if not isinstance(ut, str) or not ut.strip():
                        active[bi] = False
                        continue

                    ut = ut.strip()
                    prompt = build_prompt_with_history(histories[cid], ut, prompt_type)
                    prompts.append(prompt)
                    user_texts.append(ut)
                    idx_map.append(bi)

                if not prompts:
                    continue

                # TRUE batched generate
                outs = make_local_inference_call_batch(prompts, max_tokens=5000)

                # record + update histories
                for out_text, ut, bi in zip(outs, user_texts, idx_map):
                    cid = batch_conv_ids[bi]
                    md = batch_metadata[bi]

                    # compute turn index from column suffix (user_1 -> 0, user_2 -> 1, ...)
                    try:
                        turn_idx = int(u_col.split("_")[1]) - 1
                    except:
                        turn_idx = 0
                    rec = {
                        "conv_id": cid,
                        **md,
                        "user_turn_index": turn_idx,
                        "user_col": u_col,
                        "user_text": ut,
                        "model_output": out_text,
                    }
                    all_records.append(rec)

                    if histories[cid]:
                        histories[cid] += "\n"
                    histories[cid] += f"User: {ut}\nAI: {out_text}"

            completed_count += len(batch_conv_ids)

            # Save progress periodically (batch-safe)
            with save_lock:
                out_df = pd.DataFrame.from_records(all_records)
                out_df.to_csv(output_csv, index=False)
                print(
                    f"Progress: {completed_count}/{len(df_sub)} conversations | "
                    f"Remaining: {len(df_sub) - completed_count} | "
                    f"Total rows: {len(out_df)}"
                )

        print(f"\n✅ Done! Processed {completed_count}/{len(df_sub)} conversations")
        print(f"Final output rows: {len(all_records)}")
        print(f"Output saved to: {output_csv}\n")
        return 
    # Prepare arguments for each row (only remaining conversations)
    tasks = [
        (conv_id, row, client, model_name, user_cols, prompt_type, max_user_turns) 
        for conv_id, row in df_remaining.iterrows()
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_conv = {
            executor.submit(process_row_wrapper, task): task[0] 
            for task in tasks
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_conv):
            conv_id = future_to_conv[future]
            try:
                conv_id, records, error = future.result()
                
                if error:
                    print(f"[WARNING] conv_id={conv_id} failed: {error}")
                else:
                    all_records.extend(records)
                    completed_count += 1
                    
                    # Save progress periodically (thread-safe)
                    with save_lock:
                        out_df = pd.DataFrame.from_records(all_records)
                        out_df.to_csv(output_csv, index=False)
                        print(f"Progress: {completed_count}/{len(df_sub)} conversations | "
                              f"Remaining: {len(df_sub) - completed_count} | "
                              f"Total rows: {len(out_df)}")
                        
            except Exception as e:
                print(f"[ERROR] Unexpected error for conv_id={conv_id}: {e}")
    
    # Final save
    out_df = pd.DataFrame.from_records(all_records)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Done! Processed {completed_count}/{len(df_sub)} conversations")
    print(f"Final output rows: {len(all_records)}")
    print(f"Output saved to: {output_csv}\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        print(f"\nAvailable models: {list(AVAILABLE_MODELS.keys())}")
        print(f"Available prompt types: {list(AVAILABLE_PROMPTS.keys())}")
        sys.exit(1)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("model")
    parser.add_argument("task")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)

    args = parser.parse_args()

    api_key = args.api_key
    sample_n = args.sample_n
    max_user_turns = args.max_user_turns
    max_workers = args.max_workers
    input_csv = args.input_csv
    output_csv = args.output_csv
    model_key = args.model
    first_n = args.first_n
    prompt_type = args.task
    batch_size = args.batch_size
    print(f"Sample N: {sample_n}")
    print(f"Max user turns: {max_user_turns}")
    print(f"Max workers: {max_workers}")
    main(input_csv, output_csv, model_key, prompt_type, api_key, sample_n, first_n,max_user_turns, max_workers,batch_size)

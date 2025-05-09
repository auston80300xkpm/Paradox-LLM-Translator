import os
import yaml
import time
import json
import sqlite3
import argparse
import random
import re
import sys
import glob
import threading
import queue
import concurrent.futures
import shutil
import hashlib 
import requests 
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Set, Any 
import google.generativeai as genai
import openai 
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    InvalidArgument,
)
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm

console = Console()
CONFIG = {}

parser = argparse.ArgumentParser(
    description="Translate YML files to English using configured providers"
)
parser.add_argument("path", help="Path to YML file or folder containing YML files")
parser.add_argument(
    "--output", "-o", help="Output file or directory path", default=None
)
parser.add_argument(
    "--config",
    help="Path to YAML configuration file",
    default="config.yaml", 
)
parser.add_argument( 
    "--pattern",
    help="File pattern to match (comma-separated), overrides config if provided.",
    default=None, 
)
parser.add_argument( 
    "--threads", type=int, help="Maximum number of worker threads, overrides config if provided.", default=None
)
parser.add_argument(
    "--file-workers",
    type=int,
    help="Number of files to process in parallel",
    default=None,
)
parser.add_argument(
    "--chunk-workers", 
    type=int,
    help="Number of batches to process in parallel per file", 
    default=None,
)
parser.add_argument(
    "--max-files", type=int, help="Maximum number of files to process", default=None
)
parser.add_argument(
    "--recursive",
    "-r",
    action="store_true",
    help="Recursively process subdirectories",
)
parser.add_argument(
    "--no-ui",
    action="store_true",
    help="Disable the fancy UI and use simple console output",
)

def load_config(config_path="config.yaml"):
    global CONFIG
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        if not CONFIG:
            print(f"Error: Configuration file {config_path} is empty or invalid.")
            sys.exit(1)
        required_keys = ["target_language", "default_provider", "providers", "prompt_template", "paradox_localization", "database"]
        for key in required_keys:
            if key not in CONFIG:
                print(f"Error: Missing required key '{key}' in configuration file {config_path}.")
                sys.exit(1)
        print(f"Configuration loaded successfully from {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)

db_lock = threading.Lock()
print_lock = threading.Lock()

class UIState:
    def __init__(self):
        self.total_files = 0
        self.completed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.current_file = ""
        self.current_items_to_translate = 0 
        self.completed_items = 0 
        self.active_keys = 0
        self.total_keys = 0
        self.rate_limited_keys = 0
        self.invalid_keys = 0
        self.start_time = time.time()
        self.file_queue = []
        self.key_stats = []
        self.recent_logs = []
        self.max_logs = 10
        self.paused = False

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.recent_logs.append(f"[{timestamp}] {message}")
        if len(self.recent_logs) > self.max_logs:
            self.recent_logs.pop(0)

    def get_elapsed_time(self):
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}" # Ensure :02d for formatting

ui_state = UIState()

class TranslationDatabase:
    def __init__(self, db_path="translations.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        with db_lock:
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS file_translations (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                item_key TEXT NOT NULL, 
                original_string_hash TEXT NOT NULL, 
                original_text TEXT NOT NULL, 
                translation TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_path, item_key) 
            )
            """
            )
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS api_key_usage (
                id INTEGER PRIMARY KEY,
                api_key TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                rate_limits INTEGER DEFAULT 0,
                last_used DATETIME,
                is_valid BOOLEAN DEFAULT 1
            )
            """
            )
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS completed_files (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                output_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
            self.conn.commit()

    def save_translation(
        self, file_path: str, item_key: str, original_text: str, original_string_hash: str, translation: str
    ): 
        max_retries = 5
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                with db_lock:
                    self.cursor.execute(
                        """
                    INSERT OR REPLACE INTO file_translations (file_path, item_key, original_text, original_string_hash, translation)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                        (file_path, item_key, original_text, original_string_hash, translation),
                    )
                    self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

    def get_translation_if_hash_matches(self, file_path: str, item_key: str, original_string_hash: str) -> Optional[str]:
        with db_lock:
            self.cursor.execute(
                """
            SELECT translation FROM file_translations
            WHERE file_path = ? AND item_key = ? AND original_string_hash = ?
            """,
                (file_path, item_key, original_string_hash),
            )
            result = self.cursor.fetchone()
            return result[0] if result else None

    def mark_file_completed(self, file_path: str, output_path: str):
        with db_lock:
            self.cursor.execute(
                """
            INSERT OR REPLACE INTO completed_files (file_path, output_path)
            VALUES (?, ?)
            """,
                (file_path, output_path),
            )
            self.conn.commit()

    def is_file_completed(self, file_path: str) -> bool:
        with db_lock:
            self.cursor.execute("SELECT id FROM completed_files WHERE file_path = ?", (file_path,))
            return self.cursor.fetchone() is not None
    
    def get_completed_files(self) -> Set[str]:
        with db_lock:
            self.cursor.execute("SELECT file_path FROM completed_files")
            return {row[0] for row in self.cursor.fetchall()}

    def update_key_usage(self, api_key: str, request: bool = False, rate_limit: bool = False, valid: bool = True):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            self.cursor.execute("SELECT id FROM api_key_usage WHERE api_key = ?", (api_key,))
            if self.cursor.fetchone() is None:
                self.cursor.execute(
                    "INSERT INTO api_key_usage (api_key, requests, rate_limits, last_used, is_valid) VALUES (?, ?, ?, ?, ?)",
                    (api_key, 1 if request else 0, 1 if rate_limit else 0, now if request else None, valid)
                )
            else:
                updates, params = [], []
                if request:
                    updates.append("requests = requests + 1")
                    updates.append("last_used = ?")
                    params.append(now)
                if rate_limit: updates.append("rate_limits = rate_limits + 1")
                updates.append("is_valid = ?")
                params.extend([valid, api_key])
                self.cursor.execute(f"UPDATE api_key_usage SET {', '.join(updates)} WHERE api_key = ?", params)
            self.conn.commit()

    def get_key_usage_stats(self) -> List[Dict]:
        with db_lock:
            self.cursor.execute("SELECT api_key, requests, rate_limits, last_used, is_valid FROM api_key_usage ORDER BY requests DESC")
            return [{"api_key": r[0], "requests": r[1], "rate_limits": r[2], "last_used": r[3], "is_valid": bool(r[4])} for r in self.cursor.fetchall()]

    def close(self):
        with db_lock:
            self.conn.close()

class TranslationProvider:
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        self.config = provider_config
        self.name = provider_config.get("name", "UnknownProvider")
        self.api_keys = api_keys if api_keys else provider_config.get("api_keys", [])
        if not self.api_keys and provider_config.get("api_key"):
            self.api_keys = [provider_config.get("api_key")]
        self.db = db
        self.current_key_index = 0
        if not self.api_keys and self.config.get("requires_api_key", True):
            safe_print(f"Warning: No API keys provided for {self.name} provider and it requires keys.", console_print=False)

    def get_next_api_key(self) -> Optional[str]:
        if not self.api_keys: return None
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def translate_batch(self, texts: List[str], target_language: str, prompt_template: str, source_language: Optional[str] = None) -> List[str]:
        raise NotImplementedError("Providers must implement the translate_batch method.")
    
    def handle_rate_limit(self, api_key: str, error_message: str):
        safe_print(f"Rate limit encountered for {self.name} with key {api_key[:5]}...: {error_message}")
        if self.db and api_key: self.db.update_key_usage(api_key, rate_limit=True)

    def handle_invalid_key(self, api_key: str, error_message: str):
        safe_print(f"Invalid API key for {self.name}: {api_key[:5]}... Error: {error_message}")
        if api_key in self.api_keys:
            try:
                self.api_keys.remove(api_key)
                safe_print(f"Removed invalid key {api_key[:5]}... from {self.name} provider.")
                if self.db: self.db.update_key_usage(api_key, valid=False)
            except ValueError: pass


class GeminiProvider(TranslationProvider):
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        super().__init__(provider_config, api_keys, db)
        self.model_name = self.config.get("model", "gemini-2.5-flash-preview-04-17")

    def translate_batch(self, texts: List[str], target_language: str, prompt_template: str, source_language: Optional[str] = None) -> List[str]:
        api_key = self.get_next_api_key()
        if not api_key:
            safe_print(f"No API keys for Gemini. Batch translation failed for {len(texts)} items.")
            return texts 

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        generation_config_args = self.config.get("generation_config")
        safety_settings_args = self.config.get("safety_settings")
        
        numbered_texts_input = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(texts)])
        full_prompt = prompt_template.format(
            numbered_texts_to_translate=numbered_texts_input,
            target_language=target_language,
            source_language_name=source_language or "the original language"
        )
        
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config_args,
                safety_settings=safety_settings_args
            )
            response_content = ""
            if response.parts:
                response_content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text') and response.text:
                 response_content = response.text
            
            if response_content:
                translated_lines = response_content.strip().split('\n')
                parsed_translations = {}
                for line in translated_lines:
                    match = re.match(r"^\s*(\d+)\.\s*(.*)", line)
                    if match:
                        num = int(match.group(1))
                        translation = match.group(2).strip()
                        parsed_translations[num - 1] = translation 
                
                result = [clean_translated_text(parsed_translations.get(i, texts[i])) for i in range(len(texts))]
                if len(result) != len(texts):
                    safe_print(f"Gemini: Mismatch in translated items count. Expected {len(texts)}, got {len(result)}. Using originals for batch.")
                    return texts
                return result
            else:
                if hasattr(response, "candidates") and response.candidates and hasattr(response.candidates[0], "finish_reason"):
                    reason = response.candidates[0].finish_reason
                    if reason != 1: 
                         safe_print(f"Gemini: Batch translation stopped for reason '{reason.name}'. Originals kept.")
                else:
                    safe_print(f"Gemini: No translatable text for batch. Originals kept.")
                return texts
        except Exception as e:
            error_message = extract_error_message(e)
            safe_print(f"Gemini API error on batch with key {api_key[:5]}...: {error_message}")
            if isinstance(e, (ResourceExhausted, ServiceUnavailable)): 
                self.handle_rate_limit(api_key, error_message) 
            elif isinstance(e, InvalidArgument) or "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
                self.handle_invalid_key(api_key, error_message)
            return texts


class OpenAIProvider(TranslationProvider):
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        super().__init__(provider_config, api_keys, db)
        self.model_name = self.config.get("model", "gpt-3.5-turbo")
        self.base_url = self.config.get("base_url")
        self.client = None

    def _initialize_client(self, api_key: str):
        try:
            self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url) if self.base_url else openai.OpenAI(api_key=api_key)
        except Exception as e:
            self.client = None
            safe_print(f"Failed to initialize OpenAI client with key {api_key[:5]}...: {e}")
            raise

    def translate_batch(self, texts: List[str], target_language: str, prompt_template: str, source_language: Optional[str] = None) -> List[str]:
        numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        full_prompt = prompt_template.format(
            numbered_texts_to_translate=numbered_texts,
            target_language=target_language,
            source_language_name=source_language or "the original language"
        )
        api_key = self.get_next_api_key()
        if not api_key: return texts
        if self.client is None or (self.client and self.client.api_key != api_key):
            try: self._initialize_client(api_key)
            except Exception: self.handle_invalid_key(api_key, "Client initialization failed"); raise openai.AuthenticationError("Client initialization failed")
        if not self.client: return texts

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}], model=self.model_name
            )
            response_content = chat_completion.choices[0].message.content
            if response_content:
                translated_lines = response_content.strip().split('\n')
                parsed_translations = {}
                for line in translated_lines:
                    match = re.match(r"^\s*(\d+)\.\s*(.*)", line)
                    if match: parsed_translations[int(match.group(1)) - 1] = match.group(2).strip()
                result = [clean_translated_text(parsed_translations.get(i, texts[i])) for i in range(len(texts))]
                if len(result) != len(texts):
                    safe_print(f"OpenAI: Mismatch in translated items. Expected {len(texts)}, got {len(result)}. Using originals for batch.")
                    return texts
                return result
            else: safe_print(f"OpenAI: No translation content for batch. Key: {api_key[:5]}..."); return texts
        except openai.APIConnectionError as e: safe_print(f"OpenAI API connection error: {e}. Key: {api_key[:5]}..."); raise ServiceUnavailable(f"OpenAI connection error: {e}")
        except openai.RateLimitError as e: safe_print(f"OpenAI rate limit: {e}. Key: {api_key[:5]}..."); self.handle_rate_limit(api_key, str(e)); raise ResourceExhausted(f"OpenAI rate limit: {e}")
        except openai.AuthenticationError as e: safe_print(f"OpenAI auth error: {e}. Key: {api_key[:5]}..."); self.handle_invalid_key(api_key, str(e)); raise InvalidArgument(f"OpenAI auth error: {e}")
        except openai.APIStatusError as e:
            safe_print(f"OpenAI API status error {e.status_code}: {e.response}. Key: {api_key[:5]}...")
            if e.status_code == 401: self.handle_invalid_key(api_key, str(e.response)); raise InvalidArgument(f"OpenAI API Unauthorized: {e.response}")
            elif e.status_code == 429: self.handle_rate_limit(api_key, str(e.response)); raise ResourceExhausted(f"OpenAI API Rate Limit: {e.response}")
            elif e.status_code >= 500: raise ServiceUnavailable(f"OpenAI API Server Error: {e.response}")
            return texts
        except Exception as e: safe_print(f"Unexpected OpenAI error: {e}. Key: {api_key[:5]}..."); return texts


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        super().__init__(provider_config, api_keys, db)
        self.model_name = self.config.get("default_model", self.config.get("model", "mistralai/mistral-7b-instruct"))
        self.base_url = self.config.get("site_url", "https://openrouter.ai/api/v1")
        self.extra_headers = {"HTTP-Referer": self.config.get("your_site_url","YOUR_SITE_URL_HERE"), "X-Title": self.config.get("app_name","YML Translator")}


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        super().__init__(provider_config, api_keys, db)
        self.model_name = self.config.get("model", "deepseek-chat")
        self.base_url = self.config.get("base_url", "https://api.deepseek.com/v1")


class CustomApiProvider(TranslationProvider):
    def __init__(self, provider_config: Dict[str, Any], api_keys: List[str] = None, db: Optional[TranslationDatabase] = None):
        super().__init__(provider_config, api_keys, db)
        self.endpoint_url = self.config.get("endpoint_url")
        self.method = self.config.get("method", "POST").upper()
        self.request_payload_template = self.config.get("request_payload_template", '{}')
        self.response_translation_path = self.config.get("response_translation_path", "translation.text")
        self.headers = self.config.get("headers", {})
        self.query_params_template = self.config.get("query_params_template", {})
        self.batch_input_format = self.config.get("batch_input_format", "list_of_strings") 
        self.batch_response_format = self.config.get("batch_response_format", "list_of_strings")
        if not self.endpoint_url: raise ValueError(f"CustomApiProvider '{self.name}' missing 'endpoint_url'.")

    def _get_value_from_path(self, data: Dict, path: str) -> Optional[Any]:
        keys = path.split('.'); current = data
        for key in keys:
            if isinstance(current, dict) and key in current: current = current[key]
            elif isinstance(current, list):
                try: idx = int(key); current = current[idx] if 0 <= idx < len(current) else None
                except ValueError: return None
            else: return None
        return current

    def translate_batch(self, texts: List[str], target_language: str, prompt_template: str, source_language: Optional[str] = None) -> List[str]:
        api_key = self.get_next_api_key()
        headers = self.headers.copy()
        if api_key and self.config.get("api_key_header"): headers[self.config.get("api_key_header")] = api_key
        elif api_key and 'Authorization' not in headers: headers['Authorization'] = f'Bearer {api_key}'

        texts_for_payload = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)]) if self.batch_input_format == "numbered_string" else texts
        payload_data = {
            "texts_to_translate": texts_for_payload, "target_language": target_language,
            "source_language": source_language or "", "api_key": api_key or "",
            "prompt": prompt_template.format( 
                numbered_texts_to_translate=texts_for_payload if isinstance(texts_for_payload, str) else "\n".join(texts_for_payload),
                target_language=target_language, source_language_name=source_language or "original language"
            )}
        try:
            final_payload_str = self.request_payload_template.format(**payload_data)
            payload_json = json.loads(final_payload_str)
        except (KeyError, json.JSONDecodeError) as e: safe_print(f"Error CustomAPI payload for {self.name}: {e}. Template: {self.request_payload_template}"); return texts

        final_query_params = {}
        if self.method == "GET" and self.query_params_template:
            for k, v_template in self.query_params_template.items():
                try: final_query_params[k] = v_template.format(**payload_data)
                except KeyError as e: safe_print(f"Warning: CustomAPIProvider '{self.name}' - Missing key {e} for query param template '{k}'.")
        try:
            if self.method == "POST": response = requests.post(self.endpoint_url, json=payload_json, headers=headers, timeout=30)
            elif self.method == "GET": response = requests.get(self.endpoint_url, params=final_query_params, headers=headers, timeout=30)
            else: safe_print(f"Error: CustomAPIProvider '{self.name}' - Unsupported HTTP method '{self.method}'."); return texts
            response.raise_for_status()
            response_json = response.json()
            translated_data = self._get_value_from_path(response_json, self.response_translation_path)

            if isinstance(translated_data, list) and len(translated_data) == len(texts):
                return [clean_translated_text(str(t)) for t in translated_data]
            elif isinstance(translated_data, str) and self.batch_response_format == "numbered_string":
                translated_lines = translated_data.strip().split('\n'); parsed_translations = {}
                for line in translated_lines:
                    match = re.match(r"^\s*(\d+)\.\s*(.*)", line)
                    if match: parsed_translations[int(match.group(1)) - 1] = match.group(2).strip()
                result = [parsed_translations.get(i, texts[i]) for i in range(len(texts))]
                if len(result) != len(texts): safe_print(f"CustomAPI: Mismatch. Expected {len(texts)}, got {len(result)}."); return texts
                return [clean_translated_text(t) for t in result]
            else: safe_print(f"CustomAPI '{self.name}': Unexpected format/count. Path: '{self.response_translation_path}', Got: {translated_data}"); return texts
        except requests.exceptions.RequestException as e:
            safe_print(f"CustomAPIProvider '{self.name}' error: {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response.status_code in [401, 403] and api_key: self.handle_invalid_key(api_key, str(e.response.text))
                elif e.response.status_code == 429 and api_key: self.handle_rate_limit(api_key, str(e.response.text))
            if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)): raise ServiceUnavailable(str(e))
            return texts
        except Exception as e: safe_print(f"Unexpected error in CustomAPIProvider '{self.name}': {e}"); return texts


PROVIDER_CLASSES = {
    "gemini": GeminiProvider, "openai": OpenAIProvider, "openrouter": OpenRouterProvider,
    "deepseek": DeepSeekProvider, "custom_api": CustomApiProvider,
}

def get_translation_provider(provider_name: str, global_config: Dict[str, Any], db: Optional[TranslationDatabase]) -> Optional[TranslationProvider]:
    providers_config = global_config.get("providers")
    if not providers_config: safe_print(f"Error: 'providers' section not found in configuration."); return None
    provider_settings = providers_config.get(provider_name)
    if not provider_settings: safe_print(f"Error: Provider '{provider_name}' not found in 'providers' config."); return None
    if not provider_settings.get("enabled", False): safe_print(f"Info: Provider '{provider_name}' is not enabled."); return None
    provider_class = PROVIDER_CLASSES.get(provider_name)
    if not provider_class: safe_print(f"Error: Unknown provider type '{provider_name}'. Implemented: {list(PROVIDER_CLASSES.keys())}"); return None
    provider_settings['name'] = provider_name
    api_keys = provider_settings.get("api_keys", [])
    if not isinstance(api_keys, list): api_keys = []
    if not api_keys and provider_settings.get("api_key") and isinstance(provider_settings.get("api_key"), str):
        api_keys = [provider_settings.get("api_key")]
    if not api_keys and provider_name != "custom_api" and provider_settings.get("requires_api_key", True):
         safe_print(f"Warning: No API keys for provider '{provider_name}' and it requires keys.")
    try:
        return provider_class(provider_config=provider_settings, api_keys=api_keys, db=db)
    except Exception as e: safe_print(f"Error instantiating provider '{provider_name}': {e}"); return None

def safe_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    with print_lock:
        ui_state.add_log(message)
        if kwargs.get("console_print", True):
            print(*args, **kwargs)

class APIKeyManager: 
    def __init__(self, keys_list: List[str] = None, db: TranslationDatabase = None, base_delay: float = 1.0, rate_limit_delay: float = 30.0):
        self.keys = keys_list if keys_list is not None else []
        self.invalid_keys = set()
        self.key_data = {}
        self.base_delay = base_delay
        self.rate_limit_delay = rate_limit_delay
        self.db = db
        self.key_locks = {key: threading.Lock() for key in self.keys}
        self.global_lock = threading.Lock()
        for key in self.keys:
            self.key_data[key] = {"last_used": 0, "rate_limited_until": 0, "rate_limited_count": 0, "total_requests": 0}
        if self.db: self._load_key_status_from_db()
        ui_state.total_keys = len(self.keys)
        ui_state.active_keys = self.get_valid_keys_count()

    def _load_key_status_from_db(self):
        key_stats = self.db.get_key_usage_stats()
        for stat in key_stats:
            if stat["api_key"] in self.keys and not stat["is_valid"]:
                self.invalid_keys.add(stat["api_key"])

    def get_next_key(self) -> Optional[str]: 
        valid_keys = [k for k in self.keys if k not in self.invalid_keys]
        if not valid_keys: return None
        key = valid_keys[0] 
        return key

    def get_valid_keys_count(self) -> int: return len(self.keys) - len(self.invalid_keys)
    def get_key_stats_list(self) -> List[Dict]: 
        stats = []
        for key in self.keys:
            kd = self.key_data.get(key, {"total_requests":0, "rate_limited_count":0, "rate_limited_until":0})
            status = "INVALID" if key in self.invalid_keys else ("Rate limited" if time.time() < kd["rate_limited_until"] else "Available")
            stats.append({"key": key[:5] + "...", "requests": kd["total_requests"], "rate_limits": kd["rate_limited_count"], "status": status})
        return stats
    def get_key_stats(self) -> str: 
        return "\n".join([f"{s['key']}: Req: {s['requests']}, RL: {s['rate_limits']}, Status: {s['status']}" for s in self.get_key_stats_list()])


def get_string_hash(text: str) -> str: return hashlib.sha256(text.encode('utf-8')).hexdigest()
def extract_error_message(exception) -> str:
    error_str = str(exception)
    m = re.search(r'message: "([^"]+)"', error_str) or re.search(r'reason: "([^"]+)"', error_str)
    return m.group(1) if m else error_str
def is_rate_limit_error(error_message: str) -> bool:
    return any(p.lower() in error_message.lower() for p in ["Resource has been exhausted", "429", "quota", "rate limit", "too many requests"])

def clean_translated_text(text: str) -> str:
    if text.startswith("```yaml"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return "\n".join([line.strip() for line in text.splitlines() if line.strip() != "---"]).strip()

def process_string_value_batch(batch_data_tuple):
    items_meta, file_path, provider, db, target_language, batch_prompt_template, max_attempts = batch_data_tuple
    original_strings_in_batch = [item['original_string'] for item in items_meta]
    
    translated_strings_list = provider.translate_batch(
        texts=original_strings_in_batch, 
        target_language=target_language,
        prompt_template=batch_prompt_template, 
    )

    batch_results_map = {} 
    if len(translated_strings_list) == len(items_meta):
        for i, item_meta in enumerate(items_meta):
            key = item_meta['key']
            original_string = item_meta['original_string'] 
            original_hash = item_meta['hash']
            translated_text = translated_strings_list[i] 
            
            db.save_translation(file_path, key, original_string, original_hash, translated_text)
            batch_results_map[key] = translated_text 
            with print_lock: 
                ui_state.completed_items += 1
    else:
        safe_print(f"Error: Translation count mismatch for batch in {file_path}. Expected {len(items_meta)}, got {len(translated_strings_list)}. Using originals.")
        for item_meta in items_meta: 
            batch_results_map[item_meta['key']] = item_meta['original_string'] 
            with print_lock:
                ui_state.completed_items += 1 
            
    return batch_results_map

def get_output_path(file_path: str, output_dir: str = None, target_lang_code: str = "english") -> str:
    directory, filename = os.path.split(file_path)
    name_parts = filename.split("_l_") 
    new_filename = ""
    if len(name_parts) > 1 and len(name_parts[-1].split('.',1)[0]) > 1 : 
        base = "_l_".join(name_parts[:-1])
        old_lang_and_ext = name_parts[-1]
        ext = ""
        if '.' in old_lang_and_ext:
            ext = "." + old_lang_and_ext.split('.',1)[-1]
        new_filename = f"{base}_l_{target_lang_code}{ext}"
    else: 
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{target_lang_code}{ext}"
    return os.path.join(output_dir, new_filename) if output_dir else os.path.join(directory, new_filename)


def translate_yml_file(file_path: str, output_path: str, db: TranslationDatabase, max_workers: int, config: Dict[str, Any]) -> bool:
    if db.is_file_completed(file_path) and os.path.exists(output_path):
        safe_print(f"Skipping already translated: {file_path}")
        return True 
        
    ui_state.current_file = os.path.basename(file_path)
    file_encoding = config.get("paradox_localization", {}).get("file_encoding", "utf-8")
    try:
        with open(file_path, "r", encoding=file_encoding) as file: lines = file.readlines()
    except Exception as e: safe_print(f"Error reading {file_path}: {e}"); ui_state.failed_files += 1; return False

    paradox_regex_str = config.get("paradox_localization", {}).get("line_regex")
    if not paradox_regex_str: safe_print(f"No Paradox regex in config for {file_path}."); ui_state.failed_files += 1; return False
    try: paradox_regex = re.compile(paradox_regex_str)
    except re.error as e: safe_print(f"Error compiling Paradox regex: {e}"); ui_state.failed_files += 1; return False

    all_translatable_items_meta = []
    first_content_line_idx = 0
    if lines:
        for idx, l_text in enumerate(lines):
            if not l_text.strip().startswith("#") and l_text.strip() != "": first_content_line_idx = idx; break
        first_processing_line = first_content_line_idx + 1 if lines and re.match(r"^\s*\w+:\s*$", lines[first_content_line_idx].strip()) else 0
    else: first_processing_line = 0

    for i in range(first_processing_line, len(lines)):
        line_text = lines[i]
        match = paradox_regex.match(line_text)
        if match:
            key, original_string_value = match.group(1), match.group(2)
            all_translatable_items_meta.append({'line_idx': i, 'key': key, 'original_string': original_string_value, 
                                                'original_line_text': line_text, 'hash': get_string_hash(original_string_value)})

    if not all_translatable_items_meta:
        safe_print(f"No translatable strings found in {file_path}. Copying original.")
        if lines:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding=file_encoding) as outfile: outfile.writelines(lines)
            db.mark_file_completed(file_path, output_path); ui_state.completed_files += 1
            return True
        return False

    ui_state.current_items_to_translate = len(all_translatable_items_meta)
    ui_state.completed_items = 0
    target_language = config["target_language"]
    
    prompt_template_for_batch = config.get("string_batch_prompt_template", 
        "Translate the following texts to {target_language}. Each text is on a new line, prefixed with a number and a period (e.g., \"1. Text to translate\"). Provide your translations similarly, each on a new line, prefixed with the corresponding number and a period. Ensure the number of output translations exactly matches the number of input texts.\nInput texts:\n{numbered_texts_to_translate}\n\nTranslated texts:")

    provider_name = config.get("default_provider", "gemini")
    translation_provider = get_translation_provider(provider_name, config, db)
    if not translation_provider: safe_print(f"Failed provider init '{provider_name}' for {file_path}."); ui_state.failed_files += 1; return False
    if not translation_provider.api_keys and translation_provider.config.get("requires_api_key", True):
        safe_print(f"No API keys for provider '{translation_provider.name}' for {file_path}."); ui_state.failed_files += 1; return False
            
    max_attempts_per_batch = config.get("translation_settings", {}).get("max_translation_attempts_per_string", 3)
    
    items_needing_api_translation_meta = []
    final_translations_for_keys = {} 

    for item_meta in all_translatable_items_meta:
        cached_translation = db.get_translation_if_hash_matches(file_path, item_meta['key'], item_meta['hash'])
        if cached_translation is not None:
            final_translations_for_keys[item_meta['key']] = cached_translation
            ui_state.completed_items += 1
        else:
            items_needing_api_translation_meta.append(item_meta)
            
    safe_print(f"File {file_path}: Total items: {len(all_translatable_items_meta)}. To API: {len(items_needing_api_translation_meta)}.")

    api_batches = []
    current_batch_items_meta_for_api = []
    current_batch_char_count = 0
    max_chars_per_api_call = config.get("translation_settings", {}).get("max_chars_per_api_call", 4000)

    for item_meta in items_needing_api_translation_meta:
        item_len_estimate = len(item_meta['original_string']) + 10 
        if current_batch_char_count + item_len_estimate > max_chars_per_api_call and current_batch_items_meta_for_api:
            api_batches.append(list(current_batch_items_meta_for_api))
            current_batch_items_meta_for_api = []
            current_batch_char_count = 0
        current_batch_items_meta_for_api.append(item_meta)
        current_batch_char_count += item_len_estimate
    if current_batch_items_meta_for_api:
        api_batches.append(list(current_batch_items_meta_for_api))

    if api_batches:
        batch_processing_args_list = [
            (batch_meta_list, file_path, translation_provider, db, target_language, prompt_template_for_batch, max_attempts_per_batch)
            for batch_meta_list in api_batches
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_string_value_batch, args_tuple) for args_tuple in batch_processing_args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    translated_key_value_pairs_for_batch = future.result()
                    final_translations_for_keys.update(translated_key_value_pairs_for_batch)
                except Exception as e:
                    safe_print(f"A batch translation process failed for {file_path}: {e}")
    
    output_lines = list(lines)
    # Replace language declaration line if it was identified
    if lines and first_processing_line == first_content_line_idx + 1: # This implies the first content line was a language key
        original_lang_line = lines[first_content_line_idx].strip()
        # Construct new language line, e.g., "l_english:"
        # Ensure target_language from config is just the language name, e.g., "english"
        target_lang_name = config.get("target_language", "english").lower()
        # Preserve original leading whitespace for the language line
        leading_whitespace = lines[first_content_line_idx][:len(lines[first_content_line_idx]) - len(lines[first_content_line_idx].lstrip())]
        new_lang_line = f"{leading_whitespace}l_{target_lang_name}:\n"
        output_lines[first_content_line_idx] = new_lang_line
        safe_print(f"Replaced language line '{original_lang_line}' with '{new_lang_line.strip()}'")

    for item_meta in all_translatable_items_meta:
        line_idx, key, original_line_text = item_meta['line_idx'], item_meta['key'], item_meta['original_line_text']
        if key in final_translations_for_keys:
            translated_value = final_translations_for_keys[key]
            match = paradox_regex.match(original_line_text)
            if match:
                # Ensure original_line_text still has its newline if it had one
                original_has_newline = original_line_text.endswith('\n')
                prefix = original_line_text[:match.start(2)]
                suffix = original_line_text[match.end(2):]
                # Ensure suffix retains its newline if it was part of the original suffix
                reconstructed_line = f"{prefix}{translated_value}{suffix.rstrip(chr(10)+chr(13))}"
                if original_has_newline:
                    reconstructed_line += '\n'
                output_lines[line_idx] = reconstructed_line
            else: safe_print(f"Warning: Line {line_idx} (key: {key}) did not match regex during reconstruction.")
        
    translated_content = "".join(output_lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding=file_encoding) as file: file.write(translated_content)
    db.mark_file_completed(file_path, output_path)
    ui_state.successful_files += 1; ui_state.completed_files += 1
    safe_print(f"Translation of {file_path} to {output_path} completed.")
    return True


def translate_yml_folder(
    folder_path: str, output_dir: str, key_manager: APIKeyManager, db: TranslationDatabase,
    file_pattern: str, max_workers: int, config: Dict[str, Any], 
    max_files: int = None, recursive: bool = False,
) -> None:
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    file_paths = []
    patterns = file_pattern.split(",")
    if recursive:
        for pattern in patterns: file_paths.extend(glob.glob(os.path.join(folder_path, "**", pattern.strip()), recursive=True))
    else:
        for pattern in patterns: file_paths.extend(glob.glob(os.path.join(folder_path, pattern.strip())))

    if not file_paths: safe_print(f"No files matching pattern '{file_pattern}' found in {folder_path}"); return
    
    completed_db_files = db.get_completed_files()
    files_to_translate = []
    target_lang_code = config.get("target_language", "english").lower() 

    for fp_orig in file_paths:
        current_output_dir = output_dir
        if recursive:
            rel_path = os.path.relpath(fp_orig, folder_path)
            current_output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        
        op = get_output_path(fp_orig, current_output_dir, target_lang_code=target_lang_code)

        if fp_orig in completed_db_files and os.path.exists(op):
            safe_print(f"Skipping already translated and existing output: {fp_orig} -> {op}")
            continue
        
        files_to_translate.append((fp_orig, op)) 

    if max_files and len(files_to_translate) > max_files:
        safe_print(f"Limiting to {max_files} files out of {len(files_to_translate)} found")
        files_to_translate = files_to_translate[:max_files]

    ui_state.total_files = len(files_to_translate)
    ui_state.file_queue = [os.path.basename(f[0]) for f in files_to_translate]
    safe_print(f"Found {len(files_to_translate)} files to translate.")

    file_workers = min(max_workers, len(files_to_translate) or 1)
    batch_workers_per_file = config.get("translation_settings", {}).get("max_batch_workers_per_file", max(2, max_workers // (file_workers or 1)))
    safe_print(f"Using {file_workers} parallel file workers and up to {batch_workers_per_file} batch workers per file.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=file_workers) as executor:
        future_to_file = {
            executor.submit(translate_yml_file, fp, op, db, batch_workers_per_file, config): (fp, op)
            for fp, op in files_to_translate
        }
        for future in concurrent.futures.as_completed(future_to_file):
            fp, op = future_to_file[future]
            try:
                success = future.result()
            except Exception as e:
                safe_print(f"Error processing file {fp} in thread pool: {str(e)}")

    safe_print(f"\nProcessed {ui_state.completed_files} out of {ui_state.total_files} files initially queued.")
    safe_print(f"Successfully translated/reconstructed: {ui_state.successful_files} files.")
    safe_print(f"Failed to process: {ui_state.failed_files} files.")
    if key_manager and key_manager.keys : 
        safe_print(f"Valid keys for default provider: {key_manager.get_valid_keys_count()}/{len(key_manager.keys)}")


def create_ui_layout():
    layout = Layout()
    layout.split(Layout(name="header", size=3), Layout(name="main", ratio=1), Layout(name="footer", size=3))
    layout["main"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=1))
    layout["left"].split(Layout(name="progress", ratio=1), Layout(name="files", ratio=2))
    layout["right"].split(Layout(name="keys", ratio=1), Layout(name="logs", ratio=1))
    return layout

def update_ui(layout, live):
    header = Panel(f"[bold blue]YML Translator[/bold blue] - Running for {ui_state.get_elapsed_time()} - Press Ctrl+C to exit", style="white on blue")
    layout["header"].update(header)
    progress_table = Table.grid(expand=True)
    progress_table.add_column("Metric", style="cyan", no_wrap=True); progress_table.add_column("Value", style="green")
    progress_table.add_row("Files Processed", f"{ui_state.completed_files}/{ui_state.total_files}")
    progress_table.add_row("Successfully Translated", f"{ui_state.successful_files}")
    progress_table.add_row("Failed/Skipped", f"{ui_state.failed_files}")
    if ui_state.current_file:
        progress_table.add_row("Current File", ui_state.current_file)
        progress_table.add_row("Items", f"{ui_state.completed_items}/{ui_state.current_items_to_translate}") 
    layout["progress"].update(Panel(progress_table, title="Progress", border_style="green"))
    files_table = Table(show_header=True, header_style="bold magenta", expand=True)
    files_table.add_column("Queue Position", style="dim"); files_table.add_column("File Name", style="cyan")
    for i, file_name in enumerate(ui_state.file_queue[:10]): files_table.add_row(str(i + 1), file_name)
    if len(ui_state.file_queue) > 10: files_table.add_row("...", f"+ {len(ui_state.file_queue) - 10} more files")
    layout["files"].update(Panel(files_table, title=f"File Queue ({len(ui_state.file_queue)} remaining)", border_style="blue"))
    keys_table = Table(show_header=True, header_style="bold yellow", expand=True)
    keys_table.add_column("Key", style="dim"); keys_table.add_column("Requests", justify="right"); keys_table.add_column("Status", style="green")
    key_stats_list = ui_state.key_stats 
    for key_stat in key_stats_list:
        status_style = "red" if key_stat["status"] == "INVALID" else ("yellow" if "Rate limited" in key_stat["status"] else "green")
        keys_table.add_row(key_stat["key"], str(key_stat["requests"]), f"[{status_style}]{key_stat['status']}[/{status_style}]")
    key_summary = f"Default Provider Keys - Active: {ui_state.active_keys}/{ui_state.total_keys} | Rate Limited: {ui_state.rate_limited_keys} | Invalid: {ui_state.invalid_keys}"
    layout["keys"].update(Panel(keys_table, title=f"API Keys - {key_summary}", border_style="yellow"))
    logs_text = Text("\n".join(ui_state.recent_logs))
    layout["logs"].update(Panel(logs_text, title="Recent Logs", border_style="red"))
    footer = Panel(f"[bold]Controls:[/bold] [cyan]Ctrl+C[/cyan] to exit | Elapsed Time: {ui_state.get_elapsed_time()}", style="white on dark_blue")
    layout["footer"].update(footer)

def run_with_ui(func, *args, **kwargs):
    layout = create_ui_layout()
    thread_done = threading.Event(); result = [None]
    def thread_func():
        try: result[0] = func(*args, **kwargs)
        except Exception as e: safe_print(f"Error in main function: {str(e)}")
        finally: thread_done.set()
    thread = threading.Thread(target=thread_func, daemon=True); thread.start()
    try:
        with Live(layout, refresh_per_second=0.5, screen=True) as live:
            while not thread_done.is_set(): update_ui(layout, live); time.sleep(2)
            update_ui(layout, live) 
            console.print("\n[bold green]Processing complete! Press Enter to exit...[/bold green]"); input()
    except KeyboardInterrupt: console.print("[bold red]Interrupted. Saving progress...[/bold red]")
    return result[0]

def main():
    args = parser.parse_args()
    load_config(args.config)
    global CONFIG
    db_path = CONFIG.get("database", {}).get("path", "translations_v2.db")
    db = TranslationDatabase(db_path)
    
    default_provider_name = CONFIG.get("default_provider", "gemini")
    default_provider_config = CONFIG.get("providers", {}).get(default_provider_name, {})
    api_keys_list_for_ui_manager = default_provider_config.get("api_keys", [])
    if not api_keys_list_for_ui_manager and default_provider_config.get("api_key"):
        api_keys_list_for_ui_manager = [default_provider_config.get("api_key")]

    ui_key_manager = APIKeyManager(
        keys_list=api_keys_list_for_ui_manager, db=db,
        base_delay=CONFIG.get("translation_settings", {}).get("base_request_delay", 2.0),
        rate_limit_delay=CONFIG.get("translation_settings", {}).get("rate_limit_cool_off_delay", 30.0),
    )
    if api_keys_list_for_ui_manager:
      safe_print(f"Initialized UI KeyManager for default provider '{default_provider_name}' with {len(ui_key_manager.keys)} API keys.")
    ui_state.key_stats = ui_key_manager.get_key_stats_list()


    config_threads = CONFIG.get("translation_settings", {}).get("max_worker_threads", 4)
    threads_to_use = args.threads if args.threads is not None else config_threads
    num_file_workers = args.file_workers if args.file_workers is not None else (threads_to_use // 2 or 1)
    batch_workers_per_file = args.chunk_workers if args.chunk_workers is not None else max(2, threads_to_use // (num_file_workers or 1))
    file_pattern_to_use = args.pattern if args.pattern else CONFIG.get("paradox_localization", {}).get("file_pattern", "*.yaml,*.yml")
    target_lang_code = CONFIG.get("target_language", "english").lower()


    try:
        if os.path.isdir(args.path):
            output_dir = args.output or os.path.join(args.path, CONFIG.get("translation_settings", {}).get("default_output_folder_name", "translated"))
            translate_func_args = (args.path, output_dir, ui_key_manager, db, file_pattern_to_use, num_file_workers, CONFIG, args.max_files, args.recursive)
            if args.no_ui:
                translate_yml_folder(*translate_func_args)
            else:
                run_with_ui(translate_yml_folder, *translate_func_args)
        else: 
            output_file = args.output or get_output_path(args.path, target_lang_code=target_lang_code) 
            translate_func_args = (args.path, output_file, db, batch_workers_per_file, CONFIG)
            if args.no_ui:
                translate_yml_file(*translate_func_args)
            else:
                ui_state.total_files = 1
                ui_state.file_queue = [os.path.basename(args.path)]
                run_with_ui(translate_yml_file, *translate_func_args)
    except KeyboardInterrupt:
        safe_print("\nTranslation interrupted. Progress saved.")
    finally:
        db.close()

def show_menu(): 
    console.clear(); console.print("[bold blue]YML Translator Menu[/]", justify="center"); console.print()
    options = [("Translate (using config.yaml)", "run"), ("Exit", "exit")]
    for i, (opt, _) in enumerate(options, 1): console.print(f"[{i}] {opt}")
    choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, len(options) + 1)])
    action = options[int(choice) - 1][1]
    if action == "exit": return False
    if action == "run":
        f_path = Prompt.ask("Enter path to YML file/folder", default=CONFIG.get("default_input_path", "."))
        sys.argv = [sys.argv[0], f_path] 
        if Confirm.ask("Use UI?", default=not CONFIG.get("no_ui_default", False)):
            pass 
        else:
            sys.argv.append("--no-ui")
        main()
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    elif os.path.exists("config.yaml"):
        try:
            load_config("config.yaml")
        except SystemExit:
            pass
        
        console.print("No path argument provided. Running with `python MTL.py --help` to show options, or use interactive menu.")
        if Confirm.ask("Show interactive menu instead of help?", default=False):
            while show_menu(): pass
        else:
            parser.print_help()
    else:
        print("config.yaml not found. Please create it or specify a path with --config.")
        parser.print_help()
        sys.exit(1)

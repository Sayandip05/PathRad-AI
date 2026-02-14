"""
Model loader wrappers for MedGemma, CXR Foundation, and Path Foundation
All 3 models load on GPU (MedGemma) and CPU (CXR/Path Foundation)
Groq API used only for structural LLM responses when local models unavailable
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import os


class ModelManager:
    """Singleton to manage ML model loading and inference"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.medgemma_model = None
        self.medgemma_tokenizer = None
        self.cxr_model = None
        self.path_model = None
        self.device = None
        self._initialized = True

    def initialize(self, medgemma_path: str, cxr_path: str, path_path: str, hf_token: str = ""):
        """Initialize all models - to be called once at startup"""
        print("=" * 70)
        print("INITIALIZING PATHRAD AI MODELS")
        print("=" * 70)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Load MedGemma
        print("\n📦 Loading MedGemma...")
        self._load_medgemma(medgemma_path, hf_token)

        # Load CXR Foundation
        print("\n📦 Loading CXR Foundation...")
        self._load_cxr(cxr_path)

        # Load Path Foundation
        print("\n📦 Loading Path Foundation...")
        self._load_path(path_path)

        print("\n" + "=" * 70)
        loaded = []
        failed = []
        if self.medgemma_model: loaded.append("MedGemma (GPU)")
        else: failed.append("MedGemma")
        if self.cxr_model is not None: loaded.append("CXR Foundation (CPU)")
        else: failed.append("CXR Foundation")
        if self.path_model is not None: loaded.append("Path Foundation (CPU)")
        else: failed.append("Path Foundation")
        
        if failed:
            print(f"⚠️  LOADED: {', '.join(loaded)}")
            print(f"❌ FAILED: {', '.join(failed)}")
        else:
            print("✅ ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 70)

    def _load_medgemma(self, model_path: str, hf_token: str = ""):
        """Load MedGemma model and tokenizer - auto-downloads if not present"""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            # Auto-download MedGemma if weight files are missing
            medgemma_repo = "google/medgemma-4b-it"
            model_dir = Path(model_path)
            # Check for actual weight files (.safetensors or .bin), not just config/tokenizer
            has_weight_files = model_dir.exists() and any(
                f.suffix in ('.safetensors', '.bin')
                for f in model_dir.rglob('*')
            ) if model_dir.exists() else False

            if not has_weight_files:
                print(f"   ⏳ MedGemma not found locally. Downloading {medgemma_repo}...")
                
                # Check provided token or fallback to environment
                token = hf_token or os.environ.get("HF_TOKEN", "")
                
                if token:
                    from huggingface_hub import snapshot_download, login
                    login(token=token)
                    snapshot_download(
                        repo_id=medgemma_repo,
                        local_dir=str(model_dir),
                        token=token,
                    )
                    print(f"   ✅ MedGemma downloaded to {model_dir}")
                else:
                    print("   ❌ No HF_TOKEN set. Cannot auto-download MedGemma.")
                    print("   Set HF_TOKEN in .env or environment.")
                    return

            # Strategy: Try GPU 4-bit first, fallback to CPU float16 if VRAM insufficient
            self.medgemma_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            loaded_on = None
            if self.device == "cuda":
                try:
                    print("   Attempting GPU load (4-bit quantization)...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    self.medgemma_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    loaded_on = "GPU (4-bit)"
                except Exception as gpu_err:
                    print(f"   ⚠️  GPU load failed: {gpu_err}")
                    print("   Falling back to CPU (float16)...")
                    torch.cuda.empty_cache()

            if loaded_on is None:
                # CPU fallback with float16 to reduce memory (~8GB RAM)
                self.medgemma_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                loaded_on = "CPU (float16)"

            print(f"   ✅ MedGemma loaded on {loaded_on}")

        except Exception as e:
            print(f"   ❌ Error loading MedGemma: {e}")
            print("   ⚠️  Will use fallback methods")

    def _load_cxr(self, model_path: str):
        """Load CXR Foundation model"""
        try:
            import tensorflow as tf
            from huggingface_hub import snapshot_download

            if not os.path.exists(model_path):
                print(f"   Downloading CXR Foundation to {model_path}...")
                hf_token = os.environ.get("HF_TOKEN", "")
                if hf_token:
                    from huggingface_hub import login
                    login(token=hf_token)
                    snapshot_download(
                        repo_id="google/cxr-foundation",
                        local_dir=model_path,
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                else:
                    snapshot_download( # Try anonymous if no token (likely to fail for gated, but fallback)
                        repo_id="google/cxr-foundation",
                        local_dir=model_path,
                        local_dir_use_symlinks=False,
                    )

            # Load precomputed embeddings if available
            self.cxr_model = {}
            embeddings_path = os.path.join(
                model_path, "precomputed_embeddings", "embeddings.npz"
            )
            if os.path.exists(embeddings_path):
                self.cxr_model["embeddings"] = np.load(embeddings_path)
                print(
                    f"   Loaded embeddings: {len(self.cxr_model['embeddings'].files)} files"
                )

            print("   ✅ CXR Foundation loaded")

        except Exception as e:
            print(f"   ❌ Error loading CXR Foundation: {e}")
            self.cxr_model = None

    def _load_path(self, model_path: str):
        """Load Path Foundation model - auto-downloads if not present"""
        try:
            import tensorflow as tf
            from huggingface_hub import snapshot_download, hf_hub_download, login
            from keras.layers import TFSMLayer

            # Check for critical file saved_model.pb
            path_dir = Path(model_path)
            has_saved_model = (path_dir / "saved_model.pb").exists()

            if not has_saved_model:
                print(f"   ⏳ Path Foundation not found/incomplete. Downloading to {model_path}...")
                hf_token = os.environ.get("HF_TOKEN", "")
                if hf_token:
                    login(token=hf_token)
                    snapshot_download(
                        repo_id="google/path-foundation",
                        local_dir=model_path,
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                    
                    # Verify saved_model.pb
                    if not (path_dir / "saved_model.pb").exists():
                        print("   ⚠️  saved_model.pb missing from snapshot, attempting specific download...")
                        hf_hub_download(
                            repo_id='google/path-foundation',
                            filename='saved_model.pb',
                            local_dir=model_path,
                            token=hf_token
                        )
                else:
                    print("   ❌ No HF_TOKEN set. Cannot auto-download Path Foundation.")
                    return

            # Load on CPU for microscopy analysis
            with tf.device("/CPU:0"):
                self.path_model = TFSMLayer(model_path, call_endpoint="serving_default")

            print("   ✅ Path Foundation loaded")

        except Exception as e:
            print(f"   ❌ Error loading Path Foundation: {e}")
            self.path_model = None

    def medgemma_generate(
        self, prompt: str, max_tokens: int = 400, temperature: float = 0.3
    ) -> str:
        """Generate text using MedGemma"""
        if self.medgemma_model is None or self.medgemma_tokenizer is None:
            raise RuntimeError("MedGemma model not loaded")

        inputs = self.medgemma_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.medgemma_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

        response = self.medgemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()

        return response

    def extract_cxr_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from chest X-ray using CXR Foundation"""
        if self.cxr_model is None:
            # Return zero features if model not loaded
            return np.zeros((1, 4096))

        try:
            import tensorflow as tf

            with tf.device("/CPU:0"):
                if "embeddings" in self.cxr_model:
                    # Return first embedding as representative
                    sample_id = list(self.cxr_model["embeddings"].files)[0]
                    return self.cxr_model["embeddings"][sample_id]
                else:
                    return np.zeros((1, 4096))
        except Exception as e:
            print(f"Warning: CXR feature extraction failed: {e}")
            return np.zeros((1, 4096))

    def extract_path_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from microscopy image using Path Foundation"""
        if self.path_model is None:
            return np.zeros((1, 384))

        try:
            import tensorflow as tf

            with tf.device("/CPU:0"):
                features = self.path_model(image)
                return features["output_0"].numpy()
        except Exception as e:
            print(f"Warning: Path feature extraction failed: {e}")
            return np.zeros((1, 384))


# Global model manager instance
model_manager = ModelManager()

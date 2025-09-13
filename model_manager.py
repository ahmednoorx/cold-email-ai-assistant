import requests
import streamlit as st
import time
import shutil
import os

def download_model(model_info, models_dir="models"):
    """Download a model with progress tracking."""
    model_filename = model_info['filename']
    model_path = os.path.join(models_dir, model_filename)
    model_url = model_info['url']
    os.makedirs(models_dir, exist_ok=True)
    if os.path.exists(model_path):
        st.success(f"✅ Model {model_filename} already exists!")
        return True
    try:
        st.info(f"📥 Downloading {model_filename} (~{model_info['size_gb']}GB). This may take {model_info['download_time']}.")
        free_space = shutil.disk_usage(models_dir).free
        required_space = int(model_info['size_gb'] * 1024 * 1024 * 1024 * 1.2)
        if free_space < required_space:
            st.error(f"❌ Insufficient disk space. Need ~{model_info['size_gb']*1.2:.1f}GB free, have {free_space // (1024**3):.1f}GB")
            return False
        temp_path = model_path + ".tmp"
        if os.path.exists(temp_path):
            st.info("🧹 Cleaning up previous incomplete download...")
            os.remove(temp_path)
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing download...")
            status_text = st.empty()
            speed_text = st.empty()
        with requests.get(model_url, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(temp_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                last_time = time.time()
                last_downloaded = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        current_time = time.time()
                        if current_time - last_time > 2 or downloaded == total_size:
                            if total_size > 0:
                                percent = int(downloaded * 100 / total_size)
                                progress_bar.progress(percent, text=f"Downloading: {percent}%")
                                time_diff = current_time - last_time
                                if time_diff > 0:
                                    speed = (downloaded - last_downloaded) / time_diff
                                    speed_mb = speed / (1024 * 1024)
                                    eta_seconds = (total_size - downloaded) / speed if speed > 0 else 0
                                    eta_minutes = eta_seconds / 60
                                    status_text.text(f"📊 Progress: {downloaded//1048576}MB / {total_size//1048576}MB")
                                    speed_text.text(f"🚀 Speed: {speed_mb:.1f} MB/s • ETA: {eta_minutes:.1f} minutes")
                                last_time = current_time
                                last_downloaded = downloaded
        if total_size > 0 and downloaded != total_size:
            raise Exception(f"Download incomplete: {downloaded} bytes received, {total_size} bytes expected")
        os.rename(temp_path, model_path)
        st.success(f"✅ Model {model_filename} downloaded successfully!")
        time.sleep(2)
        return True
    except requests.exceptions.Timeout:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error("❌ Download timed out. Network connection too slow or unstable.")
        st.info("💡 **Solution:** Try again during off-peak hours or use a wired connection.")
        return False
    except requests.exceptions.ConnectionError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error("❌ Connection failed. Unable to reach download server.")
        st.info("💡 **Solution:** Check your internet connection and try again.")
        return False
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error(f"❌ Download failed: {str(e)}")
        st.info("💡 **Solution:** Check your internet connection and try again.")
        return False
"""
Model Manager for Cold Email Assistant
Handles AI model loading, system optimization, and memory management.
"""

import os
import psutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    filename: str
    size_gb: float
    ram_requirement: float
    description: str
    quality_level: str

class ModelManager:
    """Manages AI models and system resources."""
    
    def __init__(self):
        self.available_models = {}
        self.system_info = self._get_system_info()
        self._initialize_models()
    
    def _get_system_info(self) -> Dict[str, any]:
        """Get system information."""
        memory = psutil.virtual_memory()
        return {
            'ram_total_gb': round(memory.total / (1024**3), 2),
            'ram_available_gb': round(memory.available / (1024**3), 2),
            'ram_usage_percent': round((memory.total - memory.available) / memory.total * 100, 1),
            'cpu_count': psutil.cpu_count(),
            'cpu_usage_percent': psutil.cpu_percent(interval=1)
        }
    
    def _initialize_models(self):
        """Initialize available models."""
        self.available_models = {
            "mistral-7b-instruct-v0.1.Q4_K_M.gguf": ModelInfo(
                name="Mistral 7B Instruct",
                filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                size_gb=4.37,
                ram_requirement=6.0,
                description="High-quality instruction-tuned model, optimized for professional content",
                quality_level="Excellent"
            )
        }
    
    def get_model_list(self) -> List[Tuple[str, ModelInfo]]:
        """Get list of available models."""
        models = []
        for filename, info in self.available_models.items():
            model_path = os.path.join("models", filename)
            if os.path.exists(model_path):
                models.append((filename, info))
        return models
    
    def get_model_info(self, filename: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.available_models.get(filename)
    
    def is_model_compatible(self, filename: str) -> Tuple[bool, str]:
        """Check if model is compatible with current system."""
        model_info = self.get_model_info(filename)
        if not model_info:
            return False, "Model not found"
        
        model_path = os.path.join("models", filename)
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        
        # Check RAM requirement
        if self.system_info['ram_total_gb'] < model_info.ram_requirement:
            return False, f"Insufficient RAM: {model_info.ram_requirement}GB required, {self.system_info['ram_total_gb']}GB available"
        
        return True, "Compatible"
    
    def get_memory_optimized_settings(self, filename: str) -> Dict[str, any]:
        """Get memory-optimized settings for a model."""
        total_ram = self.system_info['ram_total_gb']
        current_usage = self.system_info['ram_usage_percent']
        
        # Base settings for quality
        if total_ram >= 16:
            settings = {
                'n_ctx': 4096,
                'n_batch': 512,
                'n_threads': min(8, self.system_info['cpu_count']),
                'verbose': False
            }
        elif total_ram >= 8:
            settings = {
                'n_ctx': 2048,
                'n_batch': 256,
                'n_threads': min(6, self.system_info['cpu_count']),
                'verbose': False
            }
        else:
            settings = {
                'n_ctx': 1024,
                'n_batch': 128,
                'n_threads': min(4, self.system_info['cpu_count']),
                'verbose': False
            }
        
        # Adjust for high memory usage
        if current_usage > 70:
            settings['n_ctx'] = max(512, settings['n_ctx'] // 2)
            settings['n_batch'] = max(64, settings['n_batch'] // 2)
        
        return settings
    
    def cleanup_memory(self):
        """Force memory cleanup."""
        import gc
        gc.collect()
        
        # Windows-specific memory cleanup
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except:
            pass
    
    def get_system_info(self) -> Dict[str, any]:
        """Get current system information."""
        return self._get_system_info()
    
    def get_memory_status(self) -> Dict[str, any]:
        """Get current memory status."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round((memory.total - memory.available) / (1024**3), 2),
            'usage_percent': round((memory.total - memory.available) / memory.total * 100, 1)
        }

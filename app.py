"""
Cold Email Assistant - Premium Edition
AI-powered professional email generation with multi-model support and advanced features.
"""

import streamlit as st
import pandas as pd
import time
import os
import requests
import shutil
import psutil
import platform

from datetime import datetime
from io import StringIO

# Import premium components
from email_generator import PremiumEmailGenerator, EmailVariant
from model_manager import ModelManager

# --- Session State Initialization ---
if "model_selected" not in st.session_state:
    st.session_state.model_selected = None
if "model_downloaded" not in st.session_state:
    st.session_state.model_downloaded = False

# Page configuration
st.set_page_config(
    page_title="Cold Email Assistant - Premium",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e86de);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .premium-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 10px;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .email-variant {
        border-left: 4px solid #2e86de;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'premium_generator' not in st.session_state:
    st.session_state.premium_generator = PremiumEmailGenerator()
    st.session_state.current_leads = None
    st.session_state.generated_variants = []

def detect_system_specs():
    """Detect system specifications for model recommendations."""
    try:
        # RAM detection
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # CPU detection
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_mhz = cpu_freq.current if cpu_freq else 0
        
        # Disk space detection
        disk_usage = shutil.disk_usage(".")
        free_space_gb = disk_usage.free / (1024**3)
        
        # Performance score calculation (0-100)
        performance_score = 0
        
        # RAM scoring (40% weight)
        if ram_gb >= 16:
            performance_score += 40
        elif ram_gb >= 8:
            performance_score += 30
        elif ram_gb >= 4:
            performance_score += 20
        else:
            performance_score += 10
            
        # CPU scoring (40% weight)
        if cpu_count >= 8:
            performance_score += 40
        elif cpu_count >= 4:
            performance_score += 30
        elif cpu_count >= 2:
            performance_score += 20
        else:
            performance_score += 10
            
        # Disk space scoring (20% weight)
        if free_space_gb >= 20:
            performance_score += 20
        elif free_space_gb >= 10:
            performance_score += 15
        elif free_space_gb >= 5:
            performance_score += 10
        else:
            performance_score += 5
        
        return {
            'ram_total_gb': round(ram_gb, 1),
            'ram_available_gb': round(available_ram_gb, 1),
            'cpu_count': cpu_count,
            'cpu_mhz': int(cpu_mhz),
            'free_space_gb': round(free_space_gb, 1),
            'performance_score': performance_score,
            'platform': platform.system()
        }
    except Exception as e:
        # Fallback values if detection fails
        return {
            'ram_total_gb': 8.0,
            'ram_available_gb': 4.0,
            'cpu_count': 4,
            'cpu_mhz': 2400,
            'free_space_gb': 10.0,
            'performance_score': 50,
            'platform': 'Windows'
        }

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None

def get_recommended_models(system_specs):
    """Get model recommendations based on system specifications."""
    models = []
    
    # High-performance systems
    if system_specs['performance_score'] >= 80:
        models.append({
            'name': 'Mistral 7B Instruct (Premium)',
            'filename': 'mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            'size_gb': 4.37,
            'download_time': '5-15 minutes',
            'generation_time': '45-90 seconds',
            'quality': 'Excellent',
            'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            'recommended': True,
            'description': 'Best quality for powerful systems'
        })
    
    # Medium-performance systems
    if system_specs['performance_score'] >= 50:
        models.append({
            'name': 'Mistral 7B Instruct (Balanced)',
            'filename': 'mistral-7b-instruct-v0.1.Q4_K_S.gguf',
            'size_gb': 3.52,
            'download_time': '3-10 minutes',
            'generation_time': '30-60 seconds',
            'quality': 'Very Good',
            'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf',
            'recommended': system_specs['performance_score'] < 80,
            'description': 'Good balance of speed and quality'
        })
    
    # All systems (lightweight option)
    models.append({
        'name': 'Mistral 7B Instruct (Fast)',
        'filename': 'mistral-7b-instruct-v0.1.Q4_0.gguf',
        'size_gb': 3.83,
        'download_time': '2-8 minutes',
        'generation_time': '15-30 seconds',
        'quality': 'Good',
        'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf',
        'recommended': system_specs['performance_score'] < 50,
        'description': 'Fastest generation for any system'
    })
    
    return models

def show_model_selection_screen():
    """Show the model selection screen before auto-download."""
    st.markdown("## 🚀 Welcome to Cold Email Assistant Premium!")
    st.markdown("### Let's optimize the app for your system...")
    
    # Detect system specifications
    with st.spinner("🔍 Analyzing your system..."):
        system_specs = detect_system_specs()
    
    # Show system info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💾 RAM", f"{system_specs['ram_total_gb']} GB")
    with col2:
        st.metric("🖥️ CPU Cores", system_specs['cpu_count'])
    with col3:
        st.metric("💿 Free Space", f"{system_specs['free_space_gb']} GB")
    
    # Performance assessment
    score = system_specs['performance_score']
    if score >= 80:
        st.success(f"🚀 **High Performance System** (Score: {score}/100)")
        st.info("Your system can handle the most advanced AI models with excellent speed!")
    elif score >= 50:
        st.warning(f"⚡ **Medium Performance System** (Score: {score}/100)")
        st.info("Your system works well with balanced AI models.")
    else:
        st.error(f"⚠️ **Lower Performance System** (Score: {score}/100)")
        st.info("Don't worry! We have optimized models that work great on your system.")
    
    st.markdown("---")
    st.markdown("### 🤖 Choose Your AI Model")
    st.markdown("**Select the model that best fits your needs:**")
    
    # Get model recommendations
    recommended_models = get_recommended_models(system_specs)
    
    # Show model options
    selected_model = None
    for i, model in enumerate(recommended_models):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Model card
                recommended_badge = " 🌟 **RECOMMENDED**" if model['recommended'] else ""
                st.markdown(f"""
                <div style="border: {'3px solid #28a745' if model['recommended'] else '1px solid #ddd'}; 
                           border-radius: 10px; padding: 15px; margin: 10px 0; 
                           background: {'#f8fff8' if model['recommended'] else '#f8f9fa'};">
                    <h4>{model['name']}{recommended_badge}</h4>
                    <p><strong>📊 Quality:</strong> {model['quality']}</p>
                    <p><strong>⚡ Generation Time:</strong> {model['generation_time']}</p>
                    <p><strong>📥 Download Time:</strong> {model['download_time']}</p>
                    <p><strong>💾 Size:</strong> {model['size_gb']} GB</p>
                    <p><em>{model['description']}</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Select {model['name'].split()[0]}", key=f"select_{i}", 
                           type="primary" if model['recommended'] else "secondary"):
                    selected_model = model
                    break
    
    if selected_model:
        st.session_state.selected_model = selected_model
        st.success(f"✅ Selected: {selected_model['name']}")
        st.markdown("---")

        # Confirm selection
        col1, col2 = st.columns(2)
        with col1:
            download_clicked = st.button("📥 Download & Continue", type="primary", use_container_width=True)
        with col2:
            choose_diff = st.button("🔄 Choose Different Model", use_container_width=True)

        if download_clicked:
            st.session_state.model_selection_confirmed = True
            st.session_state.start_download = True
            st.info(f"Starting download for {selected_model['name']}...")
            st.experimental_rerun()
        if choose_diff:
            if 'selected_model' in st.session_state:
                del st.session_state.selected_model
            st.experimental_rerun()

    return selected_model is not None

def download_model(model_info, models_dir="models"):
    """Download a model with progress tracking."""
    import time
    import shutil
    
    model_filename = model_info['filename']
    model_path = os.path.join(models_dir, model_filename)
    model_url = model_info['url']
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        st.success(f"✅ Model {model_filename} already exists!")
        return True
    
    try:
        st.info(f"📥 Downloading {model_filename} (~{model_info['size_gb']}GB). This may take {model_info['download_time']}.")
        
        # Check available disk space
        free_space = shutil.disk_usage(models_dir).free
        required_space = int(model_info['size_gb'] * 1024 * 1024 * 1024 * 1.2)  # 20% buffer
        
        if free_space < required_space:
            st.error(f"❌ Insufficient disk space. Need ~{model_info['size_gb']*1.2:.1f}GB free, have {free_space // (1024**3):.1f}GB")
            return False
        
        # Use temporary file to avoid corrupted partial downloads
        temp_path = model_path + ".tmp"
        
        # Clean up any existing partial downloads
        if os.path.exists(temp_path):
            st.info("🧹 Cleaning up previous incomplete download...")
            os.remove(temp_path)
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing download...")
            status_text = st.empty()
            speed_text = st.empty()
        
        # Download with progress tracking
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
                        
                        # Update progress every 2 seconds
                        current_time = time.time()
                        if current_time - last_time > 2 or downloaded == total_size:
                            if total_size > 0:
                                percent = int(downloaded * 100 / total_size)
                                progress_bar.progress(percent, text=f"Downloading: {percent}%")
                                
                                # Calculate speed
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
        
        # Verify download completeness
        if total_size > 0 and downloaded != total_size:
            raise Exception(f"Download incomplete: {downloaded} bytes received, {total_size} bytes expected")
        
        # Move temporary file to final location
        os.rename(temp_path, model_path)
        st.success(f"✅ Model {model_filename} downloaded successfully!")
        time.sleep(2)  # Brief pause to show success message
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

def cleanup_temp_files():
    """Clean up any orphaned temporary files from previous failed downloads."""
    try:
        models_dir = "models"
        if os.path.exists(models_dir):
            cleaned_count = 0
            for filename in os.listdir(models_dir):
                if filename.endswith('.tmp'):
                    temp_file_path = os.path.join(models_dir, filename)
                    try:
                        os.remove(temp_file_path)
                        cleaned_count += 1
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {filename}: {e}")
            if cleaned_count > 0:
                st.info(f"🧹 Cleaned up {cleaned_count} orphaned temporary file(s)")
    except Exception as e:
        # Silent fail - cleanup is not critical
        pass


def main():
    """Main application function."""
    st.markdown('<div class="main-header">📧 Cold Email Assistant <span class="premium-badge">PREMIUM</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Model AI Generation | Professional Quality | Unlimited Use | Privacy-First</div>', unsafe_allow_html=True)
    cleanup_temp_files()

    # --- Session State Initialization ---
    if "model_selection_confirmed" not in st.session_state:
        st.session_state.model_selection_confirmed = False
    if "model_downloaded" not in st.session_state:
        st.session_state.model_downloaded = False
    if "model_selected" not in st.session_state:
        st.session_state.model_selected = ""


    # --- Robust model loading: check file, size, and load only once ---
    model_path = os.path.join("models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}\n\nPlease ensure the ZIP was fully extracted and the model file is present.")
        st.stop()
    # File size check removed for instant testing. If model fails to load, user will see a clear error below.

    # Only load the model if not already loaded in session state
    if (
        'premium_generator' not in st.session_state
        or not hasattr(st.session_state.premium_generator, 'get_model_status')
        or not st.session_state.premium_generator.get_model_status().get('loaded', False)
    ):
        st.info(f"Loading model: {os.path.basename(model_path)}. This may take 1-2 minutes on first launch.")
        email_generator = PremiumEmailGenerator()
        try:
            load_success = email_generator.load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        except Exception as e:
            st.error(f"❌ Critical error while loading model: {e}\n\nThis is usually caused by a corrupted or incompatible model file, or insufficient system resources.\n\nTry re-downloading the model, and ensure your system meets the requirements.")
            st.stop()
        if not load_success or not email_generator.get_model_status().get('loaded', False):
            st.error(f"❌ Failed to load model: {os.path.basename(model_path)}.\n\nPossible causes:\n- Corrupted/incomplete model file\n- Incompatible llama-cpp-python version\n- Insufficient RAM or CPU\n\nPlease re-download the model and check your environment.")
            st.stop()
        st.session_state.premium_generator = email_generator
        st.success(f"✅ Model loaded: {os.path.basename(model_path)}")

    # Main app navigation (if any)
    app_mode = st.sidebar.selectbox("Choose Mode", ["✨ Single Email Generation", "📊 Bulk Email Processing", "🧪 Test & Demo"])
    if app_mode == "✨ Single Email Generation":
        single_email_generation()
    elif app_mode == "📊 Bulk Email Processing":
        bulk_email_processing()
    elif app_mode == "🧪 Test & Demo":
        demo_and_testing()

def single_email_generation():
    """Handle single email generation with variants."""
    st.header("✨ Generate Premium Email Variants")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Lead Information")
        # User hint for best results
        st.info("💡 For best results, add recent news, a product launch, or a unique fact about the recipient's company or role in the 'Additional Information' box below.")
        
        # Demo data button
        col_demo1, col_demo2 = st.columns([1, 3])
        with col_demo1:
            if st.button("🎯 Use Demo Data", help="Fill form with realistic test data"):
                demo_data = {
                    'sender_name': 'Sarah Chen',
                    'sender_company': 'DevFlow Solutions',
                    'name': 'Ahmed Ali',
                    'company': 'Devsinc',
                    'role': 'CTO',
                    'email': 'ahmed.ali@devsinc.com',
                    'industry': 'Software Development',
                    'additional_info': 'Recently launched v2.0 of their main product, focusing on microservices architecture'
                }
                st.session_state.demo_data = demo_data
                st.rerun()
        with col_demo2:
            if st.session_state.get('demo_data'):
                st.caption("✅ Demo data loaded - realistic SaaS CTO scenario")
        
        # Input form
        with st.form("email_generation_form"):
            # Get demo data if available
            demo_data = st.session_state.get('demo_data', {})
            
            # Sender Information Section
            st.subheader("👨‍💼 Your Information (Sender)")
            col_sender = st.columns(2)
            with col_sender[0]:
                sender_name = st.text_input("🏷️ Your Name *", 
                                          value=demo_data.get('sender_name', ''), 
                                          placeholder="e.g., John Smith",
                                          key="sender_name")
            with col_sender[1]:
                sender_company = st.text_input("🏢 Your Company *", 
                                             value=demo_data.get('sender_company', ''), 
                                             placeholder="e.g., TechSolutions Inc",
                                             key="sender_company")
            
            st.divider()
            
            # Recipient Information Section  
            st.subheader("🎯 Recipient Information")
            col_a, col_b = st.columns(2)
            
            with col_a:
                name = st.text_input("👤 Contact Name *", 
                                   value=demo_data.get('name', ''), 
                                   key="single_name")
                company = st.text_input("🏢 Company Name *", 
                                      value=demo_data.get('company', ''), 
                                      key="single_company")
                role = st.text_input("💼 Job Title/Role *", 
                                   value=demo_data.get('role', ''), 
                                   key="single_role")
            
            with col_b:
                email = st.text_input("📧 Email Address", 
                                    value=demo_data.get('email', ''), 
                                    key="single_email")
                industry = st.text_input("🏭 Industry", 
                                        value=demo_data.get('industry', ''), 
                                        key="single_industry")
                tone = st.selectbox("🎭 Tone", ["Professional", "Friendly", "Casual", "Direct"], key="single_tone")
            
            additional_info = st.text_area("ℹ️ Additional Information", 
                                         value=demo_data.get('additional_info', ''),
                                         placeholder="Any specific context, recent news, or details about the lead...",
                                         key="single_additional")
            
            # Premium options
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                num_variants = st.slider("📝 Number of Variants", 1, 3, 3, key="num_variants")
            with col_opt2:
                st.write("")  # Empty space to maintain layout
            
            generate_button = st.form_submit_button("🚀 Generate Email Variants", use_container_width=True)
        
        if generate_button:
            if not sender_name.strip() or not sender_company.strip():
                st.error("❌ Please fill in your name and company information.")
            elif not name.strip() or not company.strip() or not role.strip():
                st.error("❌ Please fill in the required recipient fields.")
            else:
                # Warn if pitching AI/cloud to Google or similar mismatches
                company_lower = company.strip().lower()
                role_lower = role.strip().lower()
                industry_lower = industry.strip().lower() if industry else ""
                if (
                    ("google" in company_lower and ("ai" in industry_lower or "cloud" in industry_lower or "cloud" in sender_company.lower() or "ai" in sender_company.lower()))
                    or ("openai" in company_lower and "ai" in industry_lower)
                    or ("aws" in company_lower and "cloud" in industry_lower)
                ):
                    st.warning("⚠️ The recipient's company is a leader in this field. Avoid pitching generic AI or cloud solutions to Google, OpenAI, or AWS. Make your value proposition unique and relevant!")
                # Get generation mode
                generation_mode = st.session_state.get('generation_mode', 'Standard')
                # Adjust variants based on mode
                if generation_mode == 'Express':
                    final_variants = 1
                elif generation_mode == 'Standard':
                    final_variants = min(2, num_variants)
                else:  # Premium
                    final_variants = num_variants
                generate_premium_email_variants(
                    sender_name.strip(), sender_company.strip(),
                    name.strip(), company.strip(), role.strip(), 
                    email.strip(), industry.strip(), additional_info.strip(), 
                    tone, final_variants, generation_mode
                )
    
    with col2:
        st.subheader("🎯 Generation Mode")
        
        # Generation mode selection
        mode_descriptions = {
            'Express': {
                'time': '30-60 seconds',
                'variants': '1 variant',
                'quality': 'Good',
                'description': 'Fast generation for quick results'
            },
            'Standard': {
                'time': '1-2 minutes',
                'variants': '2 variants',
                'quality': 'Very Good',
                'description': 'Balanced speed and quality'
            },
            'Premium': {
                'time': '2-3 minutes',
                'variants': '3 variants',
                'quality': 'Excellent',
                'description': 'Best quality, multiple options'
            }
        }
        
        selected_mode = st.selectbox(
            "Choose Generation Mode:",
            options=list(mode_descriptions.keys()),
            index=1,  # Default to Standard
            key="generation_mode"
        )
        
        # Show mode info
        mode_info = mode_descriptions[selected_mode]
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f8f9fa;">
            <strong>⚡ Time:</strong> {mode_info['time']}<br>
            <strong>📝 Output:</strong> {mode_info['variants']}<br>
            <strong>📊 Quality:</strong> {mode_info['quality']}<br>
            <em>{mode_info['description']}</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear form button
        if st.button("🗑️ Clear Form", use_container_width=True):
            # Clear demo data from session state
            if 'demo_data' in st.session_state:
                del st.session_state.demo_data
            st.rerun()

        st.subheader("🧠 Generation Source")
        gen_source = st.radio(
            "Select source:",
            options=["AI", "Templates"],
            index=0,
            key="generation_source",
            horizontal=True
        )
        if gen_source == "AI":
            st.caption("AI-first generation using your selected local model. Falls back to templates if no model is loaded.")
        else:
            st.caption("Template-only: fast, consistent, and safe. No AI model required.")
    
    # Display generated variants
    if st.session_state.generated_variants:
        st.header("📧 Generated Email Variants")
        
        for i, variant in enumerate(st.session_state.generated_variants, 1):
            with st.expander(f"✨ Variant {i} - Score: {variant.score}/10 ({variant.method.upper()})", expanded=i==1):
                # Variant metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Quality Score", f"{variant.score}/10")
                with col2:
                    st.metric("Generation Time", f"{variant.generation_time:.1f}s")
                with col3:
                    st.metric("Personalizations", variant.personalization_count)
                with col4:
                    st.metric("Model Used", variant.model_used)

                # Validation badge
                try:
                    lead_name = st.session_state.get('single_name', '')
                    lead_company = st.session_state.get('single_company', '')
                    lead_role = st.session_state.get('single_role', '')
                    is_valid, reasons = st.session_state.premium_generator.validate_email_variant(
                        variant,
                        name=lead_name,
                        company=lead_company,
                        role=lead_role
                    )
                    badge_color = '#16a34a' if is_valid else '#dc2626'
                    badge_text = 'Valid' if is_valid else f"Needs regen: {', '.join(reasons)}"
                    st.markdown(
                        f"<span style='display:inline-block;padding:4px 8px;border-radius:6px;background:{badge_color};color:white;font-weight:600;'>"
                        f"{badge_text}</span>",
                        unsafe_allow_html=True
                    )
                except Exception:
                    pass
                
                # Email subject and content
                st.text_input("📧 Subject Line:", variant.subject, key=f"variant_{i}_subject", disabled=True)
                st.text_area("📝 Email Content:", variant.content, height=300, key=f"variant_{i}_content")
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    # Create formatted email text for copying
                    email_text = f"Subject: {variant.subject}\n\n{variant.content}"
                    # Use code block for easy copying
                    with st.expander("📋 Copy Email", expanded=False):
                        st.code(email_text, language=None)
                        st.caption("Select all text above and copy with Ctrl+C")
                with col_b:
                    if st.button(f"🔄 Regenerate Similar", key=f"regen_{i}"):
                        st.info("Regeneration feature coming soon!")

def generate_premium_email_variants(sender_name: str, sender_company: str, 
                                   name: str, company: str, role: str, email: str, 
                                   industry: str, additional_info: str, tone: str, 
                                   num_variants: int, generation_mode: str = 'Standard'):
    """Generate premium email variants with enhanced features and time estimates."""
    
    # Validate inputs
    if not name or not company or not role:
        st.error("❌ Please provide name, company, and role")
        return
    
    if not sender_name or not sender_company:
        st.error("❌ Please provide your name and company")
        return
    
    # Time estimates based on mode
    time_estimates = {
        'Express': {'min': 30, 'max': 60},
        'Standard': {'min': 60, 'max': 120},
        'Premium': {'min': 120, 'max': 180}
    }
    
    est_time = time_estimates.get(generation_mode, time_estimates['Standard'])
    
    # Check model status but only require model if using AI generation
    model_status = st.session_state.premium_generator.get_model_status()

    # Enhanced progress feedback with time estimates
    progress_container = st.container()
    
    # Time estimate display
    with progress_container:
        st.info(f"⏱️ **{generation_mode} Mode** - Estimated time: {est_time['min']}-{est_time['max']} seconds")
        
        # Cancellation option
        col1, col2 = st.columns([3, 1])
        with col1:
            status_text = st.empty()
            progress_bar = st.progress(0)
        with col2:
            if st.button("⏹️ Cancel", key="cancel_generation"):
                st.warning("⚠️ Generation cancelled by user")
                return
    
    # Track actual generation time
    start_time = time.time()
    
    try:
        # Step 1: Initialize
        status_text.text("🔧 Initializing AI generation process...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        # Step 2: Preparing prompts
        status_text.text(f"📝 Preparing {num_variants} email prompt(s) for {generation_mode.lower()} generation...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Step 3: AI Processing with real-time updates
        status_text.text("🧠 AI is analyzing recipient and crafting personalized content...")
        progress_bar.progress(30)
        
        # Enhanced progress callback with time tracking
        def progress_callback(variant_num, total_variants, stage):
            base_progress = 30
            variant_progress = (variant_num / total_variants) * 60
            stage_progress = {"analyzing": 0, "writing": 50, "polishing": 80}.get(stage, 25)
            current_progress = min(90, base_progress + variant_progress + (stage_progress * 0.1))
            
            # Calculate elapsed time and estimate remaining
            elapsed = time.time() - start_time
            if current_progress > 30:
                estimated_total = (elapsed / (current_progress / 100))
                remaining = max(0, estimated_total - elapsed)
                time_suffix = f" (~{int(remaining)}s remaining)"
            else:
                time_suffix = ""
            
            if stage == "analyzing":
                status_text.text(f"🔍 Variant {variant_num}/{total_variants}: Analyzing recipient profile...{time_suffix}")
            elif stage == "writing":
                status_text.text(f"✍️ Variant {variant_num}/{total_variants}: Writing personalized content...{time_suffix}")
            elif stage == "polishing":
                status_text.text(f"✨ Variant {variant_num}/{total_variants}: Polishing and scoring...{time_suffix}")
            
            progress_bar.progress(int(current_progress))
        
        # Model config removed: not supported by PremiumEmailGenerator
        
        # Determine generation source (AI vs Templates)
        use_templates = st.session_state.get('generation_source', 'AI') == 'Templates'

        # If AI selected but no model loaded, warn and fall back to templates
        if not use_templates and not model_status['loaded']:
            st.warning("⚠️ No AI model loaded. Falling back to Templates.")
            use_templates = True

        # Final processing with selected generation source
        variants = st.session_state.premium_generator.generate_email_variants(
            name=name.strip(),
            company=company.strip(),
            role=role.strip(),
            industry=industry.strip() if industry else "",
            tone=tone,
            additional_info=additional_info.strip() if additional_info else "",
            num_variants=num_variants,
            sender_name=sender_name.strip(),
            sender_company=sender_company.strip(),
            progress_callback=progress_callback,
            use_templates=use_templates
        )
        
        # Step 4: Finalizing
        status_text.text("✅ Finalizing and preparing results...")
        progress_bar.progress(95)
        time.sleep(0.2)
        
        # Step 5: Complete with actual time
        actual_time = time.time() - start_time
        status_text.text(f"🎉 Generation complete in {int(actual_time)}s!")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_container.empty()
        
        if variants:
            st.session_state.generated_variants = variants
            
            # Show success with performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Generated", f"{len(variants)} variant(s)")
            with col2:
                st.metric("⚡ Time", f"{int(actual_time)}s")
            with col3:
                if actual_time <= est_time['max']:
                    st.metric("🎯 Performance", "On Time", delta="✅")
                else:
                    st.metric("🎯 Performance", "Slower", delta="⚠️")
            
            st.rerun()
        else:
            st.error("❌ Failed to generate email variants")
            
    except Exception as e:
        # Clear progress indicators on error
        progress_container.empty()
        st.error(f"❌ Generation failed: {str(e)}")
        st.info("💡 Try using Express mode for faster generation or check your system resources.")

def bulk_email_processing():
    """Handle bulk email processing."""
    st.header("📊 Bulk Email Processing")
    st.info("Upload a CSV file to generate emails for multiple leads at once.")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'], key="bulk_upload")
    
    # Demo CSV option
    col_demo1, col_demo2 = st.columns([1, 3])
    with col_demo1:
        if st.button("🎯 Load Demo CSV", help="Load sample leads for testing"):
            try:
                import os
                # Use absolute path relative to the app directory
                demo_csv_path = os.path.join(os.path.dirname(__file__), "demo_leads.csv")
                if os.path.exists(demo_csv_path):
                    st.session_state.demo_csv_data = pd.read_csv(demo_csv_path)
                    st.success("✅ Demo CSV loaded - 15 realistic B2B leads")
                    st.rerun()
                else:
                    st.error(f"Demo CSV file not found at: {demo_csv_path}")
            except Exception as e:
                st.error(f"Error loading demo CSV: {e}")
    with col_demo2:
        if st.session_state.get('demo_csv_data') is not None:
            st.caption("✅ Demo data ready - diverse tech industry leads")
    
    # Use demo data if no file uploaded but demo data loaded
    if uploaded_file:
        df_source = "uploaded file"
    elif st.session_state.get('demo_csv_data') is not None:
        uploaded_file = st.session_state.demo_csv_data
        df_source = "demo data"
    else:
        st.info("📤 Upload a CSV file or load demo data to get started.")
        return
    
    if uploaded_file is not None:
        try:
            # Handle both file upload and demo dataframe
            if hasattr(uploaded_file, 'size'):
                # File upload case
                if uploaded_file.size == 0:
                    st.error("❌ The uploaded CSV file is empty. Please upload a file with data.")
                    return
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                # Demo data case
                df = uploaded_file
            
            # Check if DataFrame is empty
            if df.empty:
                st.error(f"❌ The {df_source} contains no data rows. Please check your file format.")
                return
                
            # Check if there are any columns
            if len(df.columns) == 0:
                st.error(f"❌ The {df_source} has no columns. Please ensure it has proper headers.")
                return
                
            st.success(f"✅ Loaded {len(df)} leads from {df_source}")
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping
            st.subheader("🔗 Column Mapping")
            required_cols = ['name', 'company', 'role']
            optional_cols = ['email', 'industry', 'additional_info']
            
            col_mapping = {}
            available_columns = df.columns.tolist()
            
            # Smart auto-mapping based on column names
            auto_mapping = {}
            for col in available_columns:
                col_lower = col.lower()
                if col_lower in ['name', 'full_name', 'contact_name', 'lead_name', 'first_name']:
                    auto_mapping['name'] = col
                elif col_lower in ['company', 'company_name', 'organization', 'business']:
                    auto_mapping['company'] = col
                elif col_lower in ['role', 'position', 'title', 'job_title', 'job']:
                    auto_mapping['role'] = col
                elif col_lower in ['email', 'email_address', 'contact_email']:
                    auto_mapping['email'] = col
                elif col_lower in ['industry', 'sector', 'business_type']:
                    auto_mapping['industry'] = col
                elif col_lower in ['additional_info', 'notes', 'recent_news_snippet', 'context']:
                    auto_mapping['additional_info'] = col
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Required Fields:**")
                for col in required_cols:
                    # Use auto-mapping as default if available
                    default_idx = 0
                    if col in auto_mapping and auto_mapping[col] in available_columns:
                        default_idx = available_columns.index(auto_mapping[col])
                    col_mapping[col] = st.selectbox(f"{col.title()}:", available_columns, 
                                                   index=default_idx, key=f"map_{col}")
            
            with col2:
                st.write("**Optional Fields:**")
                for col in optional_cols:
                    # Use auto-mapping as default if available
                    options = ['None'] + available_columns
                    default_idx = 0
                    if col in auto_mapping and auto_mapping[col] in available_columns:
                        default_idx = options.index(auto_mapping[col])
                    col_mapping[col] = st.selectbox(f"{col.title()}:", options, 
                                                   index=default_idx, key=f"map_{col}")
            
            # Generation options
            st.subheader("⚙️ Generation Options")
            
            # Sender information for bulk processing
            col_sender = st.columns(2)
            with col_sender[0]:
                bulk_sender_name = st.text_input("👤 Your Name (for all emails):", 
                                                value="John Smith", 
                                                key="bulk_sender_name")
            with col_sender[1]:
                bulk_sender_company = st.text_input("🏢 Your Company (for all emails):", 
                                                   value="TechSolutions Inc", 
                                                   key="bulk_sender_company")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                bulk_tone = st.selectbox("Tone for All Emails:", ["Professional", "Friendly", "Casual", "Direct"], key="bulk_tone")
            with col_b:
                bulk_variants = st.selectbox("Variants per Lead:", [1, 2, 3], index=0, key="bulk_variants")
            with col_c:
                batch_size = st.selectbox("Batch Size:", [5, 10, 20], index=1, key="batch_size")
            
            # Generation source for bulk
            gen_source_bulk = st.radio(
                "Generation Source:",
                options=["AI", "Templates"],
                index=0,
                key="generation_source_bulk",
                horizontal=True
            )

            # Start bulk generation
            if st.button("🚀 Generate Bulk Emails", use_container_width=True):
                if not bulk_sender_name.strip() or not bulk_sender_company.strip():
                    st.error("❌ Please provide your name and company for the bulk generation.")
                else:
                    generate_bulk_emails(
                        df, col_mapping, bulk_tone, bulk_variants, batch_size,
                        bulk_sender_name.strip(), bulk_sender_company.strip(),
                        use_templates=(gen_source_bulk == "Templates")
                    )
                
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")

def generate_bulk_emails(df, col_mapping, tone, variants_per_lead, batch_size, sender_name, sender_company, use_templates=False):
    """Generate emails for all leads in bulk."""
    
    # If using AI, ensure a model is loaded; if using templates, skip model requirement
    model_status = st.session_state.premium_generator.get_model_status()
    if not use_templates and not model_status['loaded']:
        selected_model_key = st.session_state.get('model_selector', '')
        if selected_model_key:
            available_models = st.session_state.premium_generator.get_available_models()
            model_options = {}
            for filename, model_info in available_models:
                display_name = f"{model_info.name} ({model_info.quality_level})"
                model_options[display_name] = filename
            if selected_model_key in model_options:
                selected_filename = model_options[selected_model_key]
                with st.spinner(f"🤖 Loading AI model for bulk processing..."):
                    success = st.session_state.premium_generator.load_model(selected_filename)
                    if success:
                        st.success(f"✅ AI model loaded successfully!")
                    else:
                        st.error("❌ Failed to load AI model. Cannot proceed with AI bulk generation.")
                        return
            else:
                st.error("❌ No valid model selected. Please select a model first or switch to Templates.")
                return
        else:
            st.warning("⚠️ No AI model selected; proceeding with Templates for bulk generation.")
            use_templates = True
    
    results = []
    total_leads = len(df)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_leads, batch_size):
        batch_end = min(i + batch_size, total_leads)
        batch_df = df.iloc[i:batch_end]
        
        status_text.text(f"Processing leads {i+1}-{batch_end} of {total_leads}...")
        
        for idx, row in batch_df.iterrows():
            try:
                # Extract lead data using column mapping
                lead_data = {}
                for field, column in col_mapping.items():
                    # Ensure column is a string, not a method or other object
                    if column and str(column) != 'None' and str(column) in row.index:
                        lead_data[field] = str(row[str(column)]).strip()
                    else:
                        lead_data[field] = ""
                
                # Handle first_name + last_name combination
                if not lead_data.get('name') and 'first_name' in row.index and 'last_name' in row.index:
                    lead_data['name'] = f"{row['first_name']} {row['last_name']}".strip()
                
                # Debug: Check if we have required fields
                if not lead_data.get('name') or not lead_data.get('company') or not lead_data.get('role'):
                    st.warning(f"Skipping lead {idx+1}: Missing required fields (name: '{lead_data.get('name')}', company: '{lead_data.get('company')}', role: '{lead_data.get('role')}')")
                    continue
                
                # Generate variants for this lead
                variants = st.session_state.premium_generator.generate_email_variants(
                    name=lead_data.get('name', ''),
                    company=lead_data.get('company', ''),
                    role=lead_data.get('role', ''),
                    industry=lead_data.get('industry', ''),
                    tone=tone,
                    additional_info=lead_data.get('additional_info', ''),
                    num_variants=variants_per_lead,
                    sender_name=sender_name,
                    sender_company=sender_company,
                    use_templates=use_templates
                )
                
                # Add best variant to results
                if variants:
                    best_variant = variants[0]  # Already sorted by score
                    result_row = {
                        'name': lead_data.get('name', ''),
                        'company': lead_data.get('company', ''),
                        'role': lead_data.get('role', ''),
                        'email': lead_data.get('email', ''),
                        'industry': lead_data.get('industry', ''),
                        'tone': tone,
                        'generated_subject': best_variant.subject,
                        'generated_email': best_variant.content,
                        'quality_score': best_variant.score,
                        'generation_method': best_variant.method,
                        'generation_time': best_variant.generation_time,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    results.append(result_row)
                
            except Exception as e:
                st.warning(f"Failed to generate email for lead {idx+1} ({lead_data.get('name', 'Unknown')}): {str(e)}")
                continue
        
        # Update progress
        progress = (batch_end) / total_leads
        progress_bar.progress(progress)
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.success(f"✅ Generated {len(results)} emails successfully!")
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Generated", len(results))
        with col2:
            avg_score = results_df['quality_score'].mean()
            st.metric("Avg Quality Score", f"{avg_score:.1f}/10")
        with col3:
            avg_time = results_df['generation_time'].mean()
            st.metric("Avg Generation Time", f"{avg_time:.1f}s")
        with col4:
            ai_count = len(results_df[results_df['generation_method'] == 'ai'])
            st.metric("AI Generated", f"{ai_count}/{len(results)}")
        
        # Display results table
        st.subheader("📊 Generated Emails")
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"generated_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("❌ No emails were generated successfully")

def demo_and_testing():
    """Demo and testing functionality."""
    st.header("🧪 Test & Demo Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance Test")
        st.info("Test generation speed and quality with different models")
        
        if st.button("🚀 Run Performance Test", use_container_width=True):
            run_performance_test()
    
    with col2:
        st.subheader("📊 Quality Analysis")
        st.info("Analyze the quality factors of generated emails")
        
        if st.button("📈 Analyze Quality Metrics", use_container_width=True):
            analyze_quality_metrics()

def run_performance_test():
    """Run performance tests across different models."""
    st.write("🔄 Running performance tests...")
    
    available_models = st.session_state.premium_generator.get_available_models()
    if not available_models:
        st.warning("No models available for testing")
        return
    
    test_data = {
        'name': 'Sarah Johnson',
        'company': 'TechCorp Inc',
        'role': 'Chief Technology Officer',
        'industry': 'Software Technology',
        'tone': 'Professional'
    }
    
    results = []
    
    for filename, model_info in available_models:
        st.write(f"Testing {model_info.name}...")
        
        # Load model
        success = st.session_state.premium_generator.load_model(filename)
        if not success:
            continue
        
        # Generate test email
        start_time = time.time()
        try:
            variants = st.session_state.premium_generator.generate_email_variants(**test_data, num_variants=1)
            generation_time = time.time() - start_time
            
            if variants:
                variant = variants[0]
                results.append({
                    'Model': model_info.name,
                    'Quality': model_info.quality_level,
                    'Generation Time': f"{generation_time:.1f}s",
                    'Quality Score': f"{variant.score}/10",
                    'Method': variant.method
                })
        except Exception as e:
            st.error(f"Test failed for {model_info.name}: {e}")
    
    if results:
        results_df = pd.DataFrame(results)
        st.subheader("📊 Performance Test Results")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("No test results available")

def analyze_quality_metrics():
    """Analyze quality metrics of recent generations."""
    if not st.session_state.generated_variants:
        st.warning("No recent generations to analyze. Generate some emails first!")
        return
    
    st.subheader("📈 Quality Analysis")
    
    variants = st.session_state.generated_variants
    
    # Create metrics dataframe
    metrics_data = []
    for i, variant in enumerate(variants, 1):
        metrics_data.append({
            'Variant': f"Variant {i}",
            'Overall Score': variant.score,
            'Personalizations': variant.personalization_count,
            'Generation Time': variant.generation_time,
            'Method': variant.method,
            'Word Count': len(variant.content.split())
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quality Scores:**")
        st.bar_chart(metrics_df.set_index('Variant')['Overall Score'])
    
    with col2:
        st.write("**Generation Metrics:**")
        st.dataframe(metrics_df, use_container_width=True)
    
    # Best variant recommendation
    best_variant = max(variants, key=lambda x: x.score)
    st.success(f"🏆 Best Variant: {variants.index(best_variant) + 1} (Score: {best_variant.score}/10)")

if __name__ == "__main__":
    main()

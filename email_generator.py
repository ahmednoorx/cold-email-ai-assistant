"""
Cold Email Assistant - Enhanced Email Generation with Multi-Model Support
Premium version with multiple AI models, grammar correction, and quality variants.
"""

import os
import re
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# AI model imports
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None


from model_manager import ModelManager, ModelInfo
# Import the modular templates (fix import path for Windows/Python)
import sys as _sys
import os as _os
_template_path = _os.path.join(_os.path.dirname(__file__), 'src', 'templates')
if _template_path not in _sys.path:
    _sys.path.insert(0, _template_path)
import email_templates

@dataclass
class EmailVariant:
    """Container for an email variant with metadata."""
    subject: str
    content: str
    score: float
    tone: str
    method: str  # 'ai' or 'template'
    personalization_count: int
    generation_time: float
    model_used: str

class PremiumEmailGenerator:
    def _capitalize_name(self, name: str) -> str:
        """Capitalize each part of a name properly."""
        return ' '.join([part.capitalize() for part in name.split()])
    """Premium email generation engine with multi-model support and advanced features."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.current_model = None
        self.current_model_name = None
        self.model_loaded = False
        
        # Quality scoring weights
        self.scoring_weights = {
            'length': 0.15,
            'personalization': 0.25,
            'structure': 0.20,
            'grammar': 0.20,
            'call_to_action': 0.10,
            'creativity': 0.10
        }
    
    def load_model(self, model_filename: str) -> bool:
        """Load a specific AI model."""
        if not LLAMA_AVAILABLE:
            return False
        
        # Check memory before loading
        import psutil
        memory = psutil.virtual_memory()
        ram_usage_percent = ((memory.total - memory.available) / memory.total) * 100
        
        if ram_usage_percent > 85:
            print(f"⚠️ High RAM usage ({ram_usage_percent:.1f}%) - forcing cleanup before loading model")
            self.cleanup_model()
            import gc
            gc.collect()
        
        # Cleanup previous model to prevent memory leaks
        self.cleanup_model()
        
        model_info = self.model_manager.get_model_info(model_filename)
        if not model_info:
            return False
        
        # Check compatibility
        compatible, message = self.model_manager.is_model_compatible(model_filename)
        if not compatible:
            print(f"⚠️ Model not compatible: {message}")
            return False
        
        try:
            # Get memory-optimized settings to prevent RAM overuse
            settings = self.model_manager.get_memory_optimized_settings(model_filename)
            model_path = os.path.join("models", model_filename)
            
            print(f"🔧 Loading model with optimized settings: {settings}")
            
            # Force garbage collection before loading new model
            import gc
            gc.collect()
            
            # Load model with memory-optimized settings
            self.current_model = Llama(
                model_path=model_path,
                **settings
            )
            
            self.current_model_name = model_info.name
            self.model_loaded = True
            print(f"✅ Loaded model: {model_info.name}")
            
            # Force another garbage collection after loading
            gc.collect()
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model {model_info.name}: {e}")
            self.model_loaded = False
            return False
    
    # --- Quality Control Data Structures ---
    FAKE_SOCIAL_PROOF_TO_REMOVE = [
        "Stripe 40% deployment time",
        "XYZ Corp saw 30% increase",
        "Salesforce user experience optimization"
    ]

    COMPANY_SERVICE_EXCLUSIONS = {
        'Google': ['cloud services', 'AI development', 'data analytics', 'search optimization'],
        'Microsoft': ['software development', 'cloud solutions', 'productivity tools'],
        'Amazon': ['cloud services', 'e-commerce solutions', 'logistics'],
        'Salesforce': ['CRM', 'sales automation', 'customer management'],
        'OpenAI': ['AI development', 'machine learning', 'chatbot services']
    }

    ROLE_SERVICE_EXCLUSIONS = {
        'Marketing Manager': ['marketing optimization', 'digital marketing', 'campaign management'],
        'Sales Representative': ['sales tools', 'sales optimization', 'lead generation'],
        'Data Analyst': ['data analytics', 'business intelligence', 'reporting tools'],
        'Software Engineer': ['software development', 'coding tools', 'development platforms'],
        'HR Manager': ['HR software', 'recruiting tools', 'employee management']
    }

    GENERIC_SAFE_ALTERNATIVES = [
        "helped a growing company improve their processes",
        "assisted a client streamline their operations",
        "worked with a business to enhance efficiency",
        "supported an organization achieve better results"
    ]

    def _get_role_appropriate_value_prop(self, role, company):
        # Avoid pitching excluded services to roles/companies
        for exc_company, exc_services in self.COMPANY_SERVICE_EXCLUSIONS.items():
            if exc_company.lower() in company.lower():
                return f"We understand {company} is a leader in this space, so we focus on supporting your team's unique goals rather than pitching standard solutions."
        for exc_role, exc_services in self.ROLE_SERVICE_EXCLUSIONS.items():
            if exc_role.lower() in role.lower():
                return f"We help {role.lower()}s like you by providing resources and support tailored to your expertise, not generic {', '.join(exc_services)}."
        # Default generic value prop
        return "We specialize in solutions that adapt to your team's needs and help you achieve your objectives."

    def _get_generic_social_proof(self):
        return random.choice(self.GENERIC_SAFE_ALTERNATIVES)

    def _sanitize_email_content(self, content):
        # Remove any fake social proof
        for fake in self.FAKE_SOCIAL_PROOF_TO_REMOVE:
            content = content.replace(fake, "")
        # Remove any fabricated news phrases
        content = re.sub(r'recent news (item|article|about)[^\n\.!?]*[\n\.!?]', '', content, flags=re.IGNORECASE)
        content = re.sub(r'(impressive|innovative|exciting) (product launch|announcement|news)[^\n\.!?]*[\n\.!?]', '', content, flags=re.IGNORECASE)
        content = re.sub(r'(as seen in|featured in|case study|success story)[^\n\.!?]*[\n\.!?]', '', content, flags=re.IGNORECASE)
        return content

    def generate_email_variants(self, 
                               name: str, 
                               company: str, 
                               role: str, 
                               industry: str = "",
                               tone: str = "Professional",
                               additional_info: str = "",
                               num_variants: int = 1,
                               sender_name: str = "your name",
                               sender_company: str = "your company",
                               progress_callback=None,
                               use_templates: bool = False) -> List[EmailVariant]:
        """Generate multiple email variants and return the best ones."""
        # Capitalize/normalize inputs for professionalism
        name = self._capitalize_name(name)
        company = company.strip()
        sender_company = sender_company.strip()

        variants: List[EmailVariant] = []

        # Templates first (fallback if no model)
        if use_templates and num_variants > 0:
            template_count = num_variants if (not self.model_loaded or num_variants == 1) else max(1, num_variants // 2)
            template_variants = self._generate_template_variants(
                name, company, role, industry, tone, additional_info,
                template_count, sender_name, sender_company, progress_callback
            )
            variants.extend(template_variants)

        # AI variants if model is loaded and we still need more
        if self.model_loaded and len(variants) < num_variants:
            remaining = num_variants - len(variants)
            ai_variants = self._generate_ai_variants(
                name, company, role, industry, tone, additional_info, remaining,
                sender_name, sender_company, progress_callback
            )
            variants.extend(ai_variants)
        
        # Sort by score and return top variants
        variants.sort(key=lambda x: x.score, reverse=True)
        return variants[:num_variants] if variants else []
    
    def _generate_ai_variants(self, name: str, company: str, role: str, 
                             industry: str, tone: str, additional_info: str,
                             num_variants: int, sender_name: str, sender_company: str,
                             progress_callback=None) -> List[EmailVariant]:
        """Generate multiple AI-powered email variants."""
        variants: List[EmailVariant] = []
        max_retries = 3
        for i in range(num_variants):
            start_time = time.time()
            retry_count = 0
            if progress_callback:
                progress_callback(i + 1, num_variants, "analyzing")
            while retry_count < max_retries:
                try:
                    creativity_levels = [0.6, 0.7, 0.8]
                    temperature = creativity_levels[i % len(creativity_levels)]
                    prompt = self._build_ai_prompt(name, company, role, industry, tone, additional_info, sender_name, sender_company)
                    if progress_callback:
                        progress_callback(i + 1, num_variants, "writing")
                    response = self.current_model(
                        prompt,
                        max_tokens=150,  # Even more restrictive
                        temperature=0.7,  # Fixed temperature for consistency
                        top_p=0.8,
                        top_k=25,
                        repeat_penalty=1.3,  # Higher penalty
                        stop=["\n\nBest", "\n\nSincerely", "P.S.", "\n\n\n"],
                        echo=False
                    )
                    email_response = response['choices'][0]['text'].strip()
                    subject_line, email_content = self._parse_subject_and_body(email_response)
                    if progress_callback:
                        progress_callback(i + 1, num_variants, "polishing")
                    if self._is_invalid_content(email_content, subject_line):
                        retry_count += 1
                        if retry_count < max_retries:
                            continue
                        else:
                            break
                    if len(email_content) < 50 or len(subject_line) < 10:
                        retry_count += 1
                        if retry_count < max_retries:
                            continue
                        else:
                            break
                    email_content = self._clean_ai_output(email_content)
                    email_content = self._sanitize_email_content(email_content)
                    # Post-process for premium quality
                    email_content = self._scrub_cliches(email_content)
                    email_content = self._enforce_personalization(email_content, name, company, role)
                    
                    # Ultra-aggressive trimming for premium - max 80 words
                    email_content = self._trim_body_to_word_range(email_content, 50, 80)
                    email_content = self._ensure_cta(email_content)
                    
                    subject_line = self._improve_subject(subject_line, name, company, role, industry, email_content)
                    
                    # Skip PS for ultra-short emails - they should be complete without it
                    generation_time = time.time() - start_time
                    if not email_content.strip().endswith('regards,') and sender_name not in email_content:
                        if not email_content.endswith('\n'):
                            email_content += '\n'
                        email_content += f"\nBest regards,\n{sender_name}"
                    score = self._calculate_enhanced_quality_score(
                        email_content, name, company, role
                    )
                    personalization_count = self._count_personalizations(
                        email_content, name, company, role
                    )
                    variant = EmailVariant(
                        subject=subject_line,
                        content=email_content,
                        score=score,
                        tone=tone,
                        method='ai',
                        personalization_count=personalization_count,
                        generation_time=generation_time,
                        model_used=self.current_model_name or "AI Model"
                    )
                    variants.append(variant)
                    break
                except Exception:
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                    else:
                        break
        return variants

    def _generate_template_variants(self,
            name: str,
            company: str,
            role: str,
            industry: str,
            tone: str,
            additional_info: str,
            num_variants: int,
            sender_name: str,
            sender_company: str,
            progress_callback=None) -> List[EmailVariant]:
        """Generate email variants using the 5 modular templates."""
        templates = [
            lambda: email_templates.direct_approach(name, company, role, sender_name, sender_company, self._get_template_cta(tone)),
            lambda: email_templates.story_driven(name, company, role, sender_name, sender_company, self._get_template_cta(tone)),
            lambda: email_templates.problem_solution(name, company, role, sender_name, sender_company, self._get_template_cta(tone)),
            lambda: email_templates.industry_insight(name, company, role, sender_name, sender_company, self._get_template_cta(tone), industry),
            lambda: email_templates.mutual_connection(name, company, role, sender_name, sender_company, self._get_template_cta(tone)),
        ]
        variants = []
        for i in range(num_variants):
            if progress_callback:
                progress_callback(i + 1, num_variants, "writing (template)")
            template_func = random.choice(templates)
            email_content = template_func()
            subject_line, body = self._parse_subject_and_body(email_content)
            score = self._calculate_enhanced_quality_score(body, name, company, role)
            personalization_count = self._count_personalizations(body, name, company, role)
            variant = EmailVariant(
                subject=subject_line,
                content=body,
                score=score,
                tone=tone,
                method='template',
                personalization_count=personalization_count,
                generation_time=0.0,
                model_used='TEMPLATE'
            )
            variants.append(variant)
        return variants

    def _get_template_cta(self, tone: str) -> str:
        """Return a call-to-action string based on tone."""
        ctas = [
            "Would you be open to a quick call this week?",
            "Can we schedule a 15-min intro call?",
            "Are you available for a brief chat on Friday?",
            "Would you like to see a few ideas tailored to your team?",
            "Can we connect for 10 minutes next week?"
        ]
        if tone.lower() == "direct":
            return random.choice([ctas[1], ctas[2], ctas[4]])
        return random.choice(ctas)
    
    def _build_ai_prompt(self, name: str, company: str, role: str, 
                        industry: str, tone: str, additional_info: str,
                        sender_name: str = "your name", sender_company: str = "your company") -> str:
        """Build a deeply personalized, relevant prompt for AI email generation."""
        # Tone-specific instructions
        tone_map = {
            "Professional": "formal and authoritative",
            "Friendly": "warm but professional",
            "Casual": "approachable and conversational",
            "Direct": "concise and action-oriented"
        }
        tone_style = tone_map.get(tone, "professional")

        # Template/CTA variation for realism and anti-generic output
        value_prop = self._get_role_appropriate_value_prop(role, company)
        generic_social_proof = self._get_generic_social_proof()
        templates = [
            f"""
Write a professional cold email. Keep it under 80 words total.

FROM: {sender_name} at {sender_company}
TO: {name}, {role} at {company}

STRICT FORMAT:
Subject: [30-50 chars, specific business value]

Hi {name},

[1 sentence about their specific role/company context]
[1 sentence about what you offer specifically] 
[1 sentence with concrete proof/outcome]
[Clear CTA with 2 time options]

Best,
{sender_name}

Write ONLY this format:
""",
            f"""
Cold outreach email. Maximum 75 words.

TO: {name} ({role} at {company})
FROM: {sender_name} ({sender_company})

Template:
Subject: [specific hook, 30-50 chars]

Hi {name},

Brief context about {company}/role → Value prop → Social proof → CTA with times.

Best,
{sender_name}

Write the email:
""",
            f"""
Professional B2B email. Under 80 words.

{sender_name} → {name} ({role}, {company})

Subject: [business-focused, 30-50 chars]
Body: Greeting → Context → Value → Proof → CTA
Signature: Best, {sender_name}

Write it:
"""
        ]
        prompt = random.choice(templates)
        return prompt
    
    def _is_invalid_content(self, email_content: str, subject_line: str) -> bool:
        """Check if email content contains invalid meta-commentary or structural issues."""
        content_lower = email_content.lower()
        subject_lower = subject_line.lower()
        
        # Check for meta-commentary patterns
        invalid_patterns = [
            'p.s. don\'t forget',
            'remember: the key is to make',
            'be sure to research',
            'personalize your email',
            'this shows you understand',
            'email writing',
            'signature',
            'add a p.s.',
            'research your target',
            'partnership opportunity' if len(email_content) < 150 else None  # Generic subject with short content
        ]
        
        # Check for invalid patterns
        for pattern in invalid_patterns:
            if pattern and pattern in content_lower:
                return True
        
        # Check for subject line issues
        if 'p.s.' in subject_lower or 'don\'t forget' in subject_lower:
            return True
        
        # Check for lack of proper greeting
        if not any(greeting in content_lower for greeting in ['hi ', 'hello ', 'dear ', 'greetings']):
            return True
        
        # Check for lack of proper structure (no recipient name)
        if len(email_content) > 50 and not any(char.isupper() for char in email_content[:200]):
            return True
        
        return False
    
    def _calculate_enhanced_quality_score(self, email: str, name: str, company: str, role: str) -> float:
        """Calculate enhanced email quality score with stricter checks for generic, irrelevant, or poorly personalized content."""
        scores = {}
        email_lower = email.lower()
        # Length score (60-120 words optimal for premium)
        word_count = len(email.split())
        if 70 <= word_count <= 100:
            scores['length'] = 10
        elif 60 <= word_count <= 120:
            scores['length'] = 9
        elif 50 <= word_count <= 140:
            scores['length'] = 7
        else:
            scores['length'] = 3

        # Personalization score (must reference name, company, role, and at least one unique context)
        personalization_count = self._count_personalizations(email, name, company, role)
        # Penalize if only name/company/role are present (no context/news)
        if personalization_count < 3:
            scores['personalization'] = 3
        elif personalization_count == 3:
            scores['personalization'] = 6
        else:
            scores['personalization'] = min(10, personalization_count * 2)

        # Structure score
        structure_score = 0
        if any(greeting in email_lower for greeting in ['hi', 'hello', 'dear']):
            structure_score += 2
        if email.count('\n\n') >= 2:  # Multiple paragraphs
            structure_score += 3
        if any(closing in email_lower for closing in ['best', 'regards', 'sincerely']):
            structure_score += 2
        if any(cta in email_lower for cta in ['call', 'meet', 'discuss', 'schedule']):
            structure_score += 3
        scores['structure'] = structure_score

        # Call-to-action score (must be specific and time-bound)
        cta_keywords = ['call', 'meet', 'discuss', 'schedule', 'connect', 'reply', 'response', 'conversation']
        cta_score = 8 if any(keyword in email_lower for keyword in cta_keywords) and any(time_word in email_lower for time_word in ['week', 'today', 'tomorrow', 'friday', 'monday']) else 4
        scores['call_to_action'] = cta_score

        # Creativity score (variety, engagement, and avoidance of generic phrases)
        creativity_score = 5
        engaging_words = ['excited', 'impressed', 'interested', 'opportunity', 'potential', 'innovative']
        creativity_score += min(3, sum(1 for word in engaging_words if word in email_lower))
        if '?' in email:  # Questions engage readers
            creativity_score += 1
        if len(set(email_lower.split())) / (len(email.split()) or 1) > 0.7:  # Vocabulary diversity
            creativity_score += 1
        # Penalize for generic/cliche phrases and fake social proof
        generic_phrases = [
            'i hope this email finds you well',
            'as the cto at',
            'leading provider',
            'exceptional service',
            'we would be honored',
            'thank you for considering',
            'cloud optimization services',
            'ai-powered',
            'b2b sales',
            'cutting-edge',
            'innovative solutions',
            'your company',
            'our team of experts',
            'expand their customer base',
            'significant boost in revenue',
            'helping businesses like yours',
            'proven track record',
            'delivering results',
            'achieve these goals',
            'ultimate goal',
            'please visit our website',
            'give us a call at',
            'we look forward to hearing from you',
            'stripe',
            'fake client',
            'unverifiable',
            'our client',
            'as seen in',
            'featured in',
            'success story',
            'case study',
            'social proof',
        ]
        if any(phrase in email_lower for phrase in generic_phrases):
            creativity_score -= 3
        # Penalize if no context/news is referenced
        if 'recent' not in email_lower and 'announcement' not in email_lower and 'news' not in email_lower and 'launch' not in email_lower:
            creativity_score -= 1
        # Penalize if mentioning Stripe or fake clients
        if 'stripe' in email_lower or 'fake client' in email_lower or 'unverifiable' in email_lower:
            creativity_score -= 3
        # Reward if the email references the actual company or role in a non-generic way
        if company.lower() in email_lower and role.lower() in email_lower:
            creativity_score += 1
        scores['creativity'] = max(1, creativity_score)

        # Penalize if pitching AI/cloud to Google, AI to OpenAI, cloud to AWS
        if (
            ('google' in company.lower() and ('ai' in email_lower or 'cloud' in email_lower)) or
            ('openai' in company.lower() and 'ai' in email_lower) or
            ('aws' in company.lower() and 'cloud' in email_lower)
        ):
            total_score = sum(scores[key] * self.scoring_weights[key] for key in scores)
            return round(max(4.0, total_score - 2.5), 1)

        # Calculate weighted average
        total_score = sum(scores[key] * self.scoring_weights[key] for key in scores)
        return round(total_score, 1)
    
    def _count_personalizations(self, email: str, name: str, company: str, role: str) -> int:
        """Count personalization elements in the email."""
        count = 0
        
        # Ensure all inputs are strings
        email = str(email) if email else ""
        name = str(name) if name else ""
        company = str(company) if company else ""
        role = str(role) if role else ""
        
        email_lower = email.lower()
        
        if name and name.lower() in email_lower:
            count += 1
        if company and company.lower() in email_lower:
            count += 1
        if role and role.lower() in email_lower:
            count += 1
        
        # Industry and role-specific terms
        role_terms = {
            'ceo': ['leadership', 'vision', 'strategy', 'growth', 'executive'],
            'cto': ['technology', 'technical', 'development', 'innovation', 'systems'],
            'marketing': ['brand', 'campaigns', 'growth', 'audience', 'engagement'],
            'sales': ['revenue', 'targets', 'pipeline', 'customers', 'conversion'],
            'hr': ['talent', 'team', 'culture', 'recruitment', 'employee'],
            'finance': ['budget', 'costs', 'roi', 'investment', 'financial']
        }
        
        try:
            for role_key, terms in role_terms.items():
                if role_key in role.lower():
                    count += sum(1 for term in terms if term in email_lower)
                    break
        except (AttributeError, TypeError):
            # Skip role-specific terms if there's an issue
            pass
        
        return count
    
    def _clean_ai_output(self, text: str) -> str:
        """Aggressively clean AI output for premium quality."""
        # Remove common AI artifacts and meta-commentary
        text = re.sub(r'^(Here\'s|Here is).*?:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)  # Remove placeholders
        
        # Remove all forms of generic filler
        remove_patterns = [
            r'I hope this email finds you well\.?\s*',
            r'I hope this message finds you well\.?\s*',
            r'I understand the importance of.*?\.\s*',
            r"That's why it's crucial to.*?\.\s*",
            r"That's exactly what we specialize in.*?\.\s*",
            r'We would be honored to.*?\.\s*',
            r'Thank you for considering.*?\.\s*',
            r'We look forward to hearing from you.*?\.\s*',
            r'Please let us know if you.*?\.\s*',
            r'Our team of experts.*?\.\s*',
            r'We pride ourselves on.*?\.\s*'
        ]
        
        for pattern in remove_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove duplicate sentences and repetitive phrases
        sentences = text.split('. ')
        seen_sentences = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if sentence_clean and sentence_clean not in seen_sentences:
                seen_sentences.add(sentence_clean)
                unique_sentences.append(sentence.strip())
        
        text = '. '.join(unique_sentences)
        
        # Clean up spacing and formatting
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def _parse_subject_and_body(self, ai_response: str) -> Tuple[str, str]:
        """Parse subject line and email body from AI response."""
        lines = ai_response.strip().split('\n')
        subject_line = ""
        email_body = ""
        
        # Look for subject line
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('subject:'):
                # Extract subject
                subject_line = line.strip()[8:].strip()  # Remove "Subject: "
                # Get remaining lines as body
                email_body = '\n'.join(lines[i+1:]).strip()
                break
        
        # If no subject found, use first line as subject and rest as body
        if not subject_line and lines:
            if len(lines) > 1:
                subject_line = lines[0].strip()
                email_body = '\n'.join(lines[1:]).strip()
            else:
                # Generate a default subject if only body exists
                subject_line = "Partnership Opportunity"
                email_body = ai_response.strip()
        
        # Clean the subject line
        subject_line = re.sub(r'^(subject|email):\s*', '', subject_line, flags=re.IGNORECASE).strip()
        subject_line = re.sub(r'["\[\]<>]', '', subject_line).strip()
        
        # Improve generic subjects
        if subject_line.lower().startswith(('hi ', 'hello ', 'dear ', 'greetings')):
            # Generate better subject based on content
            better_subjects = [
                "Quick question about your development workflow",
                "45% efficiency boost for your team",
                "5-minute solution for CTOs",
                "Reduce development cycles by 30%",
                "Partnership opportunity for TechCorp"
            ]
            subject_line = random.choice(better_subjects)
        
        # Ensure subject is not too long
        if len(subject_line) > 60:
            subject_line = subject_line[:57] + "..."
        
        # Clean the email body
        email_body = self._clean_ai_output(email_body)
        
        return subject_line, email_body

    def _improve_subject(self, subject: str, name: str, company: str, role: str, industry: str, body: str) -> str:
        subj = (subject or '').strip()
        subj_l = subj.lower()
        too_short = len(subj) < 20
        brand_only = subj_l in {company.lower(), f"{company.lower()}:"} or subj_l.startswith(company.lower() + ' ')
        generic = any(p in subj_l for p in ['innovative solutions', 'your trusted partner', 'introduction', 'hello', 'hi'])
        if not subj or too_short or brand_only or generic:
            patterns = [
                f"Quick 15-min audit for {company}'s deployment pipeline?",
                f"One small change to reduce hotfixes at {company}",
                f"Free 10-minute CI/CD review for {company}?",
                f"{company}: cut release friction (10–15 min chat)",
                f"{role} at {company}: 3 low-lift wins this sprint"
            ]
            subj = random.choice(patterns)
        # Enforce 30–60 chars
        if len(subj) < 30:
            subj = subj + " — quick idea"
        if len(subj) > 60:
            subj = subj[:57].rstrip() + "..."
        return subj

    def _scrub_cliches(self, text: str) -> str:
        replacements = {
            r"\bleading provider\b": "practical partner",
            r"\bproven track record\b": "practical outcomes",
            r"\binnovative solutions\b": "simple, practical changes",
            r"\bexceptional innovation\b": "solid work",
            r"\btrusted partner\b": "hands-on help"
        }
        for pat, rep in replacements.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text

    def _ensure_cta(self, text: str) -> str:
        lc = text.lower()
        has_cta = any(k in lc for k in ['call', 'meet', 'schedule', 'chat', 'connect', 'reply'])
        if not has_cta:
            text = text.rstrip() + "\n\nWould Tue 10–10:30am or Wed 3–3:30pm work for a quick call?"
        return text

    def _enforce_personalization(self, body: str, name: str, company: str, role: str) -> str:
        """Ensure clean greeting and avoid repetition."""
        lines = body.strip().splitlines()
        if not lines:
            return body
        
        # Fix greeting - remove duplicate names like "Ahmed Ali Ahmed"
        first_line = lines[0].strip()
        
        # Clean up duplicate names in greeting
        name_parts = name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            # Check for patterns like "Hi Ahmed Ali Ahmed" or "Dear Ahmed Ali Ahmed"
            greeting_pattern = rf'\b(Hi|Hello|Dear)\s+({re.escape(name)})\s+({re.escape(first_name)})\b'
            first_line = re.sub(greeting_pattern, rf'\1 {name}', first_line, flags=re.IGNORECASE)
            
            # Also check for just duplicate first names
            duplicate_first_pattern = rf'\b(Hi|Hello|Dear)\s+({re.escape(first_name)})\s+({re.escape(first_name)})\b'
            first_line = re.sub(duplicate_first_pattern, rf'\1 {first_name}', first_line, flags=re.IGNORECASE)
        
        # Ensure proper greeting format
        if not first_line.lower().startswith(("hi ", "hello ", "dear ")):
            first_line = f"Hi {name},"
        elif not any(greeting in first_line.lower() for greeting in [name.lower(), name.split()[0].lower()]):
            first_line = f"Hi {name},"
        
        lines[0] = first_line
        body_text = "\n".join(lines)
        
        # Remove repetitive role/company mentions
        body_text = re.sub(rf'\bAs {role} at {company},?\s*As {role} at {company},?\s*', f'As {role} at {company}, ', body_text, flags=re.IGNORECASE)
        
        # Remove generic filler phrases
        remove_phrases = [
            r'I hope this email finds you well\.?\s*',
            r'I hope this message finds you well\.?\s*',
            r'I understand how important it is to.*?\.\s*',
            r"That's why I wanted to introduce you to.*?\.\s*",
            r'I came across your company while researching.*?\.\s*'
        ]
        
        for pattern in remove_phrases:
            body_text = re.sub(pattern, '', body_text, flags=re.IGNORECASE)
        
        return body_text

    def _trim_body_to_word_range(self, text: str, min_words: int = 80, max_words: int = 120) -> str:
        """Aggressively trim to 80-120 words max for premium quality."""
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Cut at max_words but try to preserve complete sentences
        trimmed_words = words[:max_words]
        trimmed_text = " ".join(trimmed_words)
        
        # If we cut mid-sentence, try to end at previous sentence
        if not trimmed_text.endswith(('.', '!', '?')):
            # Find last complete sentence
            sentences = trimmed_text.split('. ')
            if len(sentences) > 1:
                trimmed_text = '. '.join(sentences[:-1]) + '.'
        
        return trimmed_text

    def validate_email_variant(self, variant: EmailVariant, name: str, company: str, role: str,
                               min_subject_len: int = 30, max_subject_len: int = 60,
                               min_body_words: int = 60, max_body_words: int = 160,
                               min_score: float = 7.0) -> Tuple[bool, List[str]]:
        """Validate an EmailVariant against pragmatic acceptance rules.

        Returns (is_valid, reasons[]) where reasons contains failed rule codes.
        """
        reasons: List[str] = []
        subj = str(variant.subject or '').strip()
        body = str(variant.content or '').strip()
        
        # Ensure name, company, role are strings
        name = str(name) if name else ""
        company = str(company) if company else ""
        role = str(role) if role else ""

        # Subject length
        if len(subj) < min_subject_len or len(subj) > max_subject_len:
            reasons.append('subject_length')

        # Body length in words
        words = len(body.split())
        if words < min_body_words or words > max_body_words:
            reasons.append('body_length')

        # Personalization: require company and at least one of name/role
        lc_body = body.lower()
        personalization_hits = 0
        try:
            if company and company.lower() in lc_body:
                personalization_hits += 1
            if name and name.lower() in lc_body:
                personalization_hits += 1
            if role and role.lower() in lc_body:
                personalization_hits += 1
        except (AttributeError, TypeError):
            pass
            
        if personalization_hits < 2:
            reasons.append('personalization')

        # Placeholder scan
        placeholder_patterns = [r"\[your name\]", r"\{sender\}", r"\[company\]", r"<company>", r"YOUR COMPANY", r"YOUR NAME"]
        for pat in placeholder_patterns:
            if re.search(pat, body, flags=re.IGNORECASE):
                reasons.append('placeholder')
                break

        # Quality score threshold (recompute to reflect any changes)
        try:
            score = self._calculate_enhanced_quality_score(body, name, company, role)
        except (AttributeError, TypeError):
            score = 0.0
            
        if score < min_score:
            reasons.append('low_score')

        return (len(reasons) == 0), reasons

    def cleanup_model(self):
        """Clean up model to free memory."""
        if hasattr(self, 'current_model') and self.current_model is not None:
            try:
                # Force garbage collection of the model
                del self.current_model
                import gc
                gc.collect()
                
                # Additional memory cleanup for Windows
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                
                print("🧹 Model cleaned up to free memory")
            except Exception as e:
                print(f"⚠️ Error during model cleanup: {e}")
            finally:
                self.current_model = None
                self.model_loaded = False
                self.current_model_name = "None"
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup_model()

    def get_available_models(self) -> List[Tuple[str, ModelInfo]]:
        """Get list of available models."""
        return self.model_manager.get_model_list()
    
    def get_system_info(self) -> Dict[str, any]:
        """Get system information."""
        return self.model_manager.system_info
    
    def get_model_status(self) -> Dict[str, any]:
        """Get current model status."""
        return {
            'loaded': self.model_loaded,
            'model_name': self.current_model_name,
            'available_models': len(self.model_manager.available_models),
            'grammar_available': False,  # Grammar functionality removed
            'system_compatible': True
        }

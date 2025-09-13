"""
Email Templates Module
Each function returns a formatted email string with placeholders for personalization.
"""

def direct_approach(name, company, role, sender_name, sender_company, cta):
    return f"""
Subject: Quick Opportunity for {company}

Hi {name},

I'm reaching out directly because I believe {company} could benefit from a fresh perspective on {role}-related challenges. At {sender_company}, we've helped organizations like yours streamline operations and achieve measurable results.

If you're open to a quick chat, I'd love to share a few actionable ideas tailored to your team.

{cta}

Best regards,
{sender_name}
"""

def story_driven(name, company, role, sender_name, sender_company, cta):
    return f"""
Subject: How a {role} at {company} Made a Difference

Hi {name},

Let me share a quick story: A {role} at another company faced similar challenges to what {company} is tackling now. By rethinking their approach, they saw significant improvements in efficiency and morale. At {sender_company}, we specialize in helping teams like yours turn obstacles into opportunities.

Would you be interested in hearing more?

{cta}

Sincerely,
{sender_name}
"""

def problem_solution(name, company, role, sender_name, sender_company, cta):
    return f"""
Subject: Solving {role} Challenges at {company}

Hi {name},

Many {role}s at companies like {company} struggle with time-consuming processes and limited resources. Our team at {sender_company} has developed solutions that address these exact pain points, leading to better outcomes and less stress.

Let's connect to discuss how we can help you overcome these challenges.

{cta}

Best,
{sender_name}
"""

def industry_insight(name, company, role, sender_name, sender_company, cta, industry):
    return f"""
Subject: {industry} Trends Impacting {company}

Hi {name},

The {industry} landscape is evolving rapidly, and {company} is well-positioned to take advantage of new opportunities. At {sender_company}, we keep a close eye on industry trends and help leaders like you stay ahead of the curve.

Would you like to see a few insights relevant to your team?

{cta}

Regards,
{sender_name}
"""

def mutual_connection(name, company, role, sender_name, sender_company, cta, connection_name=None):
    intro = f"I was recently speaking with {connection_name}, who spoke highly of your work at {company}." if connection_name else f"I've heard great things about {company} and your leadership as a {role}."
    return f"""
Subject: Introduction via Our Network

Hi {name},

{intro}

At {sender_company}, we love connecting with forward-thinking professionals. I believe there's potential for us to collaborate or share ideas that could benefit your team.

{cta}

Warm regards,
{sender_name}
"""

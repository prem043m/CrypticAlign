import random
import pandas as pd
from faker import Faker

fake = Faker("en_IN")

# -----------------------------
# CONFIG
# -----------------------------

NUM_USERS = 300

LOCATIONS = [
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Kolkata",
    "Ahmedabad"
]

PROFESSIONS = [
    "Data Scientist",
    "ML Engineer",
    "AI Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "Product Manager",
    "Project Manager",
    "Consultant",
    "Financial Analyst",
    "Investment Advisor",
    "Accountant",
    "Doctor",
    "Nurse",
    "Healthcare Analyst",
    "UI/UX Designer",
    "Graphic Designer",
    "Content Writer",
    "Marketing Specialist"
]

PROFESSION_GROUPS = {
    "TECH": [
        "Data Scientist",
        "ML Engineer",
        "AI Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "DevOps Engineer",
        "Cloud Engineer",
        "Cybersecurity Analyst"
    ],

    "BUSINESS": [
        "Business Analyst",
        "Product Manager",
        "Project Manager",
        "Consultant"
    ],

    "FINANCE": [
        "Financial Analyst",
        "Investment Advisor",
        "Accountant"
    ],

    "HEALTHCARE": [
        "Doctor",
        "Nurse",
        "Healthcare Analyst"
    ],

    "CREATIVE": [
        "UI/UX Designer",
        "Graphic Designer",
        "Content Writer",
        "Marketing Specialist"
    ]
}

# Enhanced Education by Group
EDUCATION_BY_GROUP = {
    "TECH": [
        "B.Tech Computer Science",
        "B.Tech IT",
        "MCA",
        "M.Tech AI",
        "B.Tech Electronics",
        "M.Tech Data Science"
    ],

    "BUSINESS": [
        "MBA",
        "BBA",
        "Management Studies",
        "B.Com",
        "MBA Operations"
    ],

    "FINANCE": [
        "B.Com",
        "CA",
        "MBA Finance",
        "CFA",
        "B.Sc Economics"
    ],

    "HEALTHCARE": [
        "MBBS",
        "BDS",
        "B.Sc Nursing",
        "MPH",
        "M.Sc Public Health"
    ],

    "CREATIVE": [
        "BFA Design",
        "Fine Arts",
        "Mass Communication",
        "Graphic Design Diploma",
        "UX Certification"
    ]
}

# Networking Intents
NETWORKING_INTENTS = [
    "Find Mentor",
    "Find Mentee",
    "Career Growth",
    "Startup Partner",
    "Professional Networking",
    "Research Collaboration",
    "Team Building",
    "Knowledge Sharing"
]

# Personality Traits
PERSONALITY_TRAITS = [
    "Leadership",
    "Creative",
    "Analytical",
    "Collaborative",
    "Innovative",
    "Detail-Oriented",
    "Strategic",
    "Adaptable",
    "Problem Solver",
    "Communication",
    "Empathetic",
    "Visionary"
]

INTERESTS = [
    "AI",
    "Machine Learning",
    "Fitness",
    "Reading",
    "Teaching",
    "Startups",
    "Travel",
    "Photography",
    "Gaming",
    "Music",
    "Finance",
    "Healthcare",
    "Public Speaking",
    "Mentoring",
    "Writing",
    "Data Science"
]

PROFESSION_SKILLS = {
    "Data Scientist": [
        "Python", "Machine Learning",
        "SQL", "Statistics",
        "Data Analysis", "Tableau",
        "A/B Testing"
    ],

    "ML Engineer": [
        "Python",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "AWS",
        "Model Deployment",
        "Computer Vision"
    ],

    "AI Engineer": [
        "Python",
        "LLMs",
        "NLP",
        "Deep Learning",
        "Vector Databases",
        "Prompt Engineering",
        "Transformers"
    ],

    "Backend Developer": [
        "Java",
        "SQL",
        "Docker",
        "API Development",
        "System Design",
        "Microservices",
        "MongoDB"
    ],

    "Frontend Developer": [
        "React",
        "JavaScript",
        "HTML",
        "CSS",
        "UI Design",
        "TypeScript",
        "Redux"
    ],

    "DevOps Engineer": [
        "Docker",
        "Kubernetes",
        "AWS",
        "CI/CD",
        "Linux",
        "Terraform",
        "Jenkins"
    ],

    "Cloud Engineer": [
        "AWS",
        "Azure",
        "Cloud Architecture",
        "Infrastructure",
        "DevOps",
        "Networking",
        "Security"
    ],

    "Cybersecurity Analyst": [
        "Cybersecurity",
        "Network Security",
        "SIEM",
        "Threat Analysis",
        "Linux",
        "Penetration Testing",
        "Incident Response"
    ],

    "Product Manager": [
        "Product Strategy",
        "Data Analysis",
        "Leadership",
        "User Research",
        "Roadmap Planning",
        "Analytics"
    ],

    "Doctor": [
        "Patient Care",
        "Diagnosis",
        "Clinical Research",
        "Healthcare Management",
        "Medical Documentation"
    ],

    "Financial Analyst": [
        "Finance",
        "Excel",
        "Risk Analysis",
        "Forecasting",
        "Business Intelligence",
        "Financial Modeling"
    ],

    "UI/UX Designer": [
        "UI Design",
        "UX Research",
        "Figma",
        "Prototyping",
        "User Testing",
        "Design Systems"
    ]
}

PROFESSION_INTERESTS = {
    "Data Scientist": [
        "AI",
        "Machine Learning",
        "Data Science",
        "Reading",
        "Teaching"
    ],

    "ML Engineer": [
        "AI",
        "Machine Learning",
        "Startups",
        "Data Science",
        "Gaming"
    ],

    "AI Engineer": [
        "AI",
        "Machine Learning",
        "Startups",
        "Gaming",
        "Research"
    ],

    "Doctor": [
        "Healthcare",
        "Teaching",
        "Reading",
        "Fitness",
        "Mentoring"
    ],

    "Financial Analyst": [
        "Finance",
        "Public Speaking",
        "Reading",
        "Writing",
        "Teaching"
    ],

    "Graphic Designer": [
        "Photography",
        "Music",
        "Travel",
        "Writing",
        "Design"
    ],

    "Product Manager": [
        "Startups",
        "Leadership",
        "Technology",
        "Reading",
        "Mentoring"
    ]
}

ALL_MBTI = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]

MBTI_MAPPING = {
    "Data Scientist": ["INTJ","INTP","ISTJ"],
    "ML Engineer": ["INTJ","INTP","ENTJ"],
    "AI Engineer": ["INTJ","INTP","ENTJ"],
    "Product Manager": ["ENTJ","ENFJ","ESTJ"],
    "Project Manager": ["ENTJ","ESTJ"],
    "Graphic Designer": ["INFP","ENFP","ISFP"],
    "Content Writer": ["INFP","ENFP"],
    "UI/UX Designer": ["INFP","ENFP","ISFP"],
    "Doctor": ["ISFJ","INFJ","ESFJ"]
}

CAREER_GOAL_WEIGHTS = {
    "Data Scientist": {
        "AI Research": 45,
        "Data Analytics": 40,
        "Leadership": 15
    },

    "ML Engineer": {
        "AI Research": 50,
        "Cloud Computing": 35,
        "Leadership": 15
    },

    "AI Engineer": {
        "AI Research": 60,
        "Startup Founder": 40
    },

    "Backend Developer": {
        "Cloud Computing": 55,
        "Leadership": 20,
        "Startup Founder": 25
    },

    "Frontend Developer": {
        "Startup Founder": 40,
        "Leadership": 35,
        "Product Management": 25
    },

    "Cybersecurity Analyst": {
        "Cybersecurity": 70,
        "Leadership": 30
    },

    "Financial Analyst": {
        "Financial Growth": 70,
        "Leadership": 30
    },

    "Doctor": {
        "Healthcare Innovation": 70,
        "Leadership": 30
    },

    "Product Manager": {
        "Startup Founder": 50,
        "Leadership": 50
    },

    "UI/UX Designer": {
        "Startup Founder": 40,
        "Leadership": 35,
        "Design Leadership": 25
    }
}

CAREER_GOALS = [
    "Leadership",
    "Startup Founder",
    "AI Research",
    "Data Analytics",
    "Cloud Computing",
    "Cybersecurity",
    "Product Management",
    "Financial Growth",
    "Healthcare Innovation",
    "Design Leadership"
]
# -----------------------------
# TEXT TEMPLATES
# -----------------------------

SUMMARY_TEMPLATES = [
    """{profession} with {experience} years of experience in the industry.
Skilled in {skill1}, {skill2}, and {skill3}.
Passionate about innovation and continuous learning.
Education: {education}""",

    """Experienced {profession} specializing in modern solutions.
Strong expertise in {skill1} and {skill2}.
Interested in building impactful products.
Education: {education}""",

    """Dedicated {profession} with {experience} years of hands-on experience.
Expert in {skill1}, {skill2}, {skill3}.
Seeking {networking_intent} opportunities.
Education: {education}"""
]

ABOUT_TEMPLATES = [
    """I enjoy solving real-world problems and collaborating with teams.
Outside work I enjoy {interest1} and {interest2}.
Personality: {trait1}, {trait2}.
Networking Goal: {networking_intent}""",

    """Curious and creative thinker who values lifelong learning.
I enjoy {interest1}, {interest2}, and mentoring others.
Traits: {trait1}, {trait2}, {trait3}.
Looking for: {networking_intent}""",

    """Passionate about making an impact in {career_goal}.
I'm {trait1} and {trait2}.
Interested in {interest1}, {interest2}.
Networking Intent: {networking_intent}"""
]

SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Power BI",
    "Java",
    "AWS",
    "Docker",
    "Leadership",
    "Communication",
    "Data Analysis",
    "Cybersecurity"
]


# -----------------------------
# USER GENERATOR
# -----------------------------

def generate_user(user_num):

    profession = random.choice(PROFESSIONS)

    # MBTI Assignment
    if profession in MBTI_MAPPING:
        mbti = random.choice(
            MBTI_MAPPING[profession]
        )
    else:
        mbti = random.choice(
            ALL_MBTI
        )

    # Profession Based Interests
    if profession in PROFESSION_INTERESTS:

        available_interests = \
            PROFESSION_INTERESTS[profession]

        interests = random.sample(
            available_interests,
            min(
                len(available_interests),
                random.randint(3, 5)
            )
        )

    else:

        interests = random.sample(
            INTERESTS,
            random.randint(3, 5)
        )

    # Profession Based Skills
    if profession in PROFESSION_SKILLS:

        available_skills = \
            PROFESSION_SKILLS[profession]

        skills = random.sample(
            available_skills,
            min(
                len(available_skills),
                3
            )
        )

    else:

        skills = random.sample(
            SKILLS,
            3
        )

    # Career Goal Mapping
    if profession in CAREER_GOAL_WEIGHTS:

        goals = list(
            CAREER_GOAL_WEIGHTS[profession].keys()
        )

        weights = list(
            CAREER_GOAL_WEIGHTS[profession].values()
        )

        career_goal = random.choices(
            goals,
            weights=weights,
            k=1
        )[0]
    else:
        career_goal = random.choice(
            CAREER_GOALS
        )
        
    experience = random.randint(0, 20)

    age = max(
        20,
        min(
            45,
            experience + random.randint(22, 30)
        )
    )

    # NEW: Education based on profession
    group = None
    for g, profs in PROFESSION_GROUPS.items():
        if profession in profs:
            group = g
            break
    if group is None:
        group = "TECH"
    
    education = random.choice(
        EDUCATION_BY_GROUP.get(group, EDUCATION_BY_GROUP["TECH"])
    )
    
    # NEW: Personality Traits
    num_traits = random.randint(2, 3)
    traits = random.sample(PERSONALITY_TRAITS, num_traits)
    
    # NEW: Networking Intent
    networking_intent = random.choice(NETWORKING_INTENTS)

    # Professional Summary (Enhanced)
    summary = random.choice(
        SUMMARY_TEMPLATES
    ).format(
        profession=profession,
        experience=experience,
        skill1=skills[0],
        skill2=skills[1],
        skill3=skills[2],
        education=education,
        networking_intent=networking_intent
    )

    # About Me (Enhanced)
    about = random.choice(
        ABOUT_TEMPLATES
    ).format(
        interest1=interests[0],
        interest2=interests[1],
        trait1=traits[0],
        trait2=traits[1],
        trait3=traits[2] if len(traits) > 2 else traits[1],
        networking_intent=networking_intent,
        career_goal=career_goal
    )

    return {

        "user_id":
            f"U{user_num:03}",

        "name":
            fake.name(),

        "age":
            age,

        "location":
            random.choice(
                LOCATIONS
            ),

        "profession":
            profession,

        "experience_years":
            experience,

        "education":
            education,

        "skills":
            ",".join(skills),

        "mbti":
            mbti,

        "traits":
            ",".join(traits),

        "career_goal":
            career_goal,

        "networking_intent":
            networking_intent,

        "interests":
            ",".join(interests),

        "professional_summary":
            summary,

        "about_me":
            about
    }

def generate_users():

    users = []

    for i in range(1, NUM_USERS + 1):
        users.append(generate_user(i))

    users_df = pd.DataFrame(users)

    users_df.to_csv(
        "../users.csv",
        index=False
    )

    return users_df


if __name__ == "__main__":
    print("Generating synthetic user dataset...")
    df = generate_users()
    print(f"[SUCCESS] Generated {len(df)} users in ../users.csv")
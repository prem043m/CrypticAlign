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
        "Data Analysis"
    ],

    "ML Engineer": [
        "Python",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "AWS"
    ],

    "AI Engineer": [
        "Python",
        "LLMs",
        "NLP",
        "Deep Learning",
        "Vector Databases"
    ],

    "Backend Developer": [
        "Java",
        "SQL",
        "Docker",
        "API Development",
        "System Design"
    ],

    "Frontend Developer": [
        "React",
        "JavaScript",
        "HTML",
        "CSS",
        "UI Design"
    ],

    "Cybersecurity Analyst": [
        "Cybersecurity",
        "Network Security",
        "SIEM",
        "Threat Analysis",
        "Linux"
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
        "Business Intelligence"
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
        "Data Science"
    ],

    "AI Engineer": [
        "AI",
        "Machine Learning",
        "Startups",
        "Gaming"
    ],

    "Doctor": [
        "Healthcare",
        "Teaching",
        "Reading",
        "Fitness"
    ],

    "Financial Analyst": [
        "Finance",
        "Public Speaking",
        "Reading",
        "Writing"
    ],

    "Graphic Designer": [
        "Photography",
        "Music",
        "Travel",
        "Writing"
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
    "UI/UX Designer": ["INFP","ENFP","ISFP"]
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
    "Healthcare Innovation"
]
# -----------------------------
# TEXT TEMPLATES
# -----------------------------

SUMMARY_TEMPLATES = [
    """{profession} with {experience} years of experience in the industry.
Skilled in {skill1}, {skill2}, and {skill3}.
Passionate about innovation and continuous learning.""",

    """Experienced {profession} specializing in modern solutions.
Strong expertise in {skill1} and {skill2}.
Interested in building impactful products."""
]

ABOUT_TEMPLATES = [
    """I enjoy solving real-world problems and collaborating with teams.
Outside work I enjoy {interest1} and {interest2}.""",

    """Curious and creative thinker who values lifelong learning.
I enjoy {interest1}, {interest2}, and mentoring others."""
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

    # Professional Summary
    summary = random.choice(
        SUMMARY_TEMPLATES
    ).format(
        profession=profession,
        experience=experience,
        skill1=skills[0],
        skill2=skills[1],
        skill3=skills[2]
    )

    summary += (
        f"\nCareer Goal: {career_goal}."
    )

    # About Me
    about = random.choice(
        ABOUT_TEMPLATES
    ).format(
        interest1=interests[0],
        interest2=interests[1]
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

        "career_goal":
            career_goal,

        "professional_summary":
            summary,

        "about_me":
            about,

        "mbti":
            mbti,

        "interests":
            ",".join(interests)
    }

def generate_users():

    users = []

    for i in range(1, NUM_USERS + 1):
        users.append(generate_user(i))

    users_df = pd.DataFrame(users)

    users_df.to_csv(
        "users.csv",
        index=False
    )

    return users_df
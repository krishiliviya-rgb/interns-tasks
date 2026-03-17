from model import rank_resumes
from utils import load_resumes

# -----------------------------------
# Load Job Description
# -----------------------------------

with open("job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

# -----------------------------------
# Load All Resumes
# -----------------------------------

resumes, filenames = load_resumes("resumes")

# -----------------------------------
# Rank Resumes Using ML Model
# -----------------------------------

scores = rank_resumes(job_description, resumes)

results = list(zip(filenames, scores))

# Sort by highest match score
results.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 Candidate Ranking:\n")

for name, score in results:
    print(f"{name} --> Match Score: {round(score * 100, 2)}%")

# -----------------------------------
# Missing Skills Detection
# -----------------------------------

print("\n🔎 Missing Skills Analysis:\n")

# Define important skills (based on job role)
important_skills = [
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "scikit-learn",
    "tensorflow",
    "data preprocessing",
    "feature engineering"
]

for i, resume in enumerate(resumes):

    resume_lower = resume.lower()
    missing = []

    for skill in important_skills:
        if skill not in resume_lower:
            missing.append(skill)

    print(f"{filenames[i]} missing skills:")
    print(missing)
    print()
import os

def load_resumes(folder_path):
    resumes = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                resumes.append(f.read())
                filenames.append(file)

    return resumes, filenames
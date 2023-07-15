import os
import glob

REPO_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TEMPLATE_PATH = REPO_PATH + "/.github/TEMPLATE.md"
README_PATH = REPO_PATH + "/README.md"

print("Parsing experiments...")
experiments = []
folder_paths = glob.glob(REPO_PATH+'/**/')
folder_paths = [path for path in folder_paths if os.path.isdir(path)]
for path in folder_paths:
    experiment_name = os.path.basename(os.path.dirname(path))
    print("-", experiment_name)
    experiments.append(experiment_name)
print("Done!")

print("Parsing descriptions...")
descriptions = []
for experiment_name in experiments:

    with open(REPO_PATH + "/" + experiment_name + "/README.md", 'r') as experiment_readme_file:
        idx = 1
        lines = experiment_readme_file.readlines()
        if lines[idx].strip() == "": idx+=1
        description = lines[idx]
        description = description.strip()

    print("-", experiment_name, ":", description)
    descriptions.append(description)
print("Done!")

print("Reading the template...")
with open(TEMPLATE_PATH, 'r') as template_file:
    template_contents = template_file.readlines()
print("Done!")

print("Building the table...")
template_contents.append("| Experiment | Description |\n")
template_contents.append("| ---------- | ----------- |\n")
for experiment_name, description in zip(experiments,descriptions):
    template_contents.append("| [" + experiment_name + "](" + experiment_name + "/README.md) | " + description + " |\n")
print(template_contents)
print("Done!")

print("Writing contents to README.md...")
with open(README_PATH, 'w') as readme_file:
    readme_file.writelines(template_contents)
print("Done!")


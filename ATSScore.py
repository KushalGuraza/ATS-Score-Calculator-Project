# %%
# Importing Libs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from docx import Document
import PyPDF2

# %%
# Function removes special characters and converts the text to lowercase
def preprocess_text(text):
    # Remove special characters and numbers and replacing with space " "
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    return text.lower()

# %%
def get_resume_score(resume, job_description):
    # Preprocess the texts
    resume = preprocess_text(resume)
    job_description = preprocess_text(job_description)

    # Combine resume and job description in a list
    text = [resume, job_description]

    # Create CountVectorizer object
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text)

    # Calculate cosine similarity
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

# %%
def read_file(folder_path, file_name):
    docx_path = os.path.join(folder_path, f"{file_name}.docx")
    pdf_path = os.path.join(folder_path, f"{file_name}.pdf")

    if os.path.exists(docx_path):
        doc = Document(docx_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in reader.pages])
    else:
        raise FileNotFoundError(f"No DOCX or PDF file found for {file_name}")

# %%
folder_path = 'C:\\Users\kusha\Kushal Guraza\OneDrive\Desktop\PROJECTS\ATSScoreProject'
try:
    resume = read_file(folder_path, 'resume')
    job_description = read_file(folder_path, 'JD')
    print("Files read successfully")
except FileNotFoundError as e:
    print(f"Error: {e}")

# %%
score = get_resume_score(resume, job_description)
print(f"Your resume matches about {score}% of the job description.")

# %%
#end

# %%


# %%


# %%








# %%


# %%




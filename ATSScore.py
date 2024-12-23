
# Importing Libs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from docx import Document
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


# Function removes special characters and converts the text to lowercase
def preprocess_text(text):

    # Filter out short lines
    lines = text.splitlines()
    text = ' '.join([line for line in lines if len(line.split()) > 2])

    # Text normalization - Lowercasing, Removing special characters and numbers, replacing with space " " and removing extra spaces
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(text)

def get_resume_score_and_missing_keywords(resume, job_description):

    # Preprocess the texts
    resume = preprocess_text(resume)
    job_description = preprocess_text(job_description)
    # print("\n" + resume + "\n")
    # print("\n" + job_description + "\n")

    # splitting the texts into set of unique keywords
    resume_words = set(resume.split())
    job_description_words = set(job_description.split())

    # Identifying missing keywords
    missing_keywords = job_description_words - resume_words

    # Combine resume and job description in a list
    text = [resume, job_description]

    # Create CountVectorizer object
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)

    # Calculate cosine similarity
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100

    # TF-IDF Vectorization to capture word importance
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(text)

    # Compute cosine similarity between resume and job description
    # match_percentage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    return round(match_percentage, 2), missing_keywords


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

folder_path = r'C:\Users\kusha\Kushal Guraza\OneDrive\Desktop\PROJECTS\ATSScoreProject'

try:
    # Read the resume and job description files
    resume = read_file(folder_path, 'resume')
    job_description = read_file(folder_path, 'JD')
    print("Files read successfully")

    # Get resume score and missing keywords
    score, missing_keywords = get_resume_score_and_missing_keywords(resume, job_description)
    print(f"Your resume matches about {score}% of the job description.")

    if missing_keywords:
        print("\nKeywords missing from your resume")
        print(",".join(missing_keywords))
    else:
        print("\nYour resume contains all the Keywords from the Job description")

except FileNotFoundError as e:
    print(f"Error: {e}")


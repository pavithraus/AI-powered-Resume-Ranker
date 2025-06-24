# AI-Powered Resume Ranker System - Enhanced Version
# Includes improved ranking logic, refined keyword matching, and additional feedback features

import os
import io
import json
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import PyPDF2
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import Counter
import zipfile
from fuzzywuzzy import fuzz
from dateutil.relativedelta import relativedelta
from dateutil import parser
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install SpaCy English model: py -m spacy download en_core_web_sm")
    nlp = None

class ResumeRanker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85
        )
        self.job_description = ""
        self.resumes = []
        self.scores = []

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def preprocess_text(self, text):
        if not nlp:
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            return text

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return ' '.join(tokens)

    def extract_keywords(self, text, top_n=25):
        if not nlp:
            words = text.lower().split()
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_n)]

        doc = nlp(text)
        keywords = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT']]
        keywords += [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 2]
        keyword_freq = Counter(keywords)
        return [word for word, _ in keyword_freq.most_common(top_n)]

    def calculate_keyword_match_score(self, resume_text, job_keywords):
        resume_words = re.findall(r'\b\w+\b', resume_text.lower())
        resume_words = set(word for word in resume_words if word not in STOP_WORDS)
        job_words = set(word.lower() for word in job_keywords if word.lower() not in STOP_WORDS)
        matches = len(resume_words.intersection(job_words))
        return (matches / len(job_words)) * 100 if job_words else 0



    def calculate_experience_score(self, resume_text):
     resume_text = resume_text.lower()
     date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s?\d{2,4})\s*(?:to|–|-)\s*(present|current|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s?\d{2,4})',
        resume_text,
        flags=re.IGNORECASE
     )

     total_years = 0
     for start_str, end_str in date_ranges:
        try:
            start_date = parser.parse(start_str, fuzzy=True, default=datetime(2000, 1, 1))
            if 'present' in end_str or 'current' in end_str:
                end_date = datetime.now()
            else:
                end_date = parser.parse(end_str, fuzzy=True, default=datetime(2000, 1, 1))
            diff = relativedelta(end_date, start_date)
            years = diff.years + (diff.months / 12)
            total_years += years
        except Exception as e:
            continue  # skip if can't parse

     return min(int(total_years * 10), 100)


 
    def calculate_education_score(self, resume_text):
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'mba', 'degree',
            'university', 'college', 'institute', 'certification',
            'certified', 'diploma', 'graduate','bsc','msc','mca'
        ]
        matches = sum(1 for keyword in education_keywords if keyword in resume_text.lower())
        return min(matches * 15, 100)

    def calculate_skills_score(self, resume_text, job_description, tech_skills=None):
     if tech_skills is None:
            tech_skills = [
     'python', 'r', 'sql', 'java', 'c++','c', 'ml', 'ai', 'machine learning',
     'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
     'keras', 'xgboost', 'lightgbm', 'pandas', 'numpy', 'scikit-learn',
     'matplotlib', 'seaborn', 'plotly', 'excel', 'tableau', 'power bi',
     'data visualization', 'data analysis', 'statistics', 'data storytelling',
      'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'mlflow',
     'spark', 'hadoop', 'mysql', 'postgresql', 'mongodb', 'airflow', 'kafka',
     'vscode', 'jupyter', 'colab', 'agile', 'scrum', 'teamwork',
     'problem-solving', 'presentation', 'google analytics', 'transformers','data handling','cloud',
     'big data']
     job_lower = job_description.lower()
     resume_lower = resume_text.lower()

     # Extract skills from job description using fuzzy matching
     job_skills = []
     for skill in tech_skills:
        if skill in job_lower:
            job_skills.append((skill, 1.0))  # Full weight for exact match
        elif fuzz.partial_ratio(skill, job_lower) > 80:
            job_skills.append((skill, 0.7))  # Partial match weight

     if not job_skills:
        return 50  # Neutral if no skills in job description

     total_weight = 0
     matched_weight = 0

     for skill, weight in job_skills:
        total_weight += weight
        if skill in resume_lower or fuzz.partial_ratio(skill, resume_lower) > 80:
            matched_weight += weight

     return (matched_weight / total_weight) * 100 if total_weight > 0 else 0


    def rank_resumes(self, job_description, resume_files):
        self.job_description = job_description
        self.resumes = []
        self.scores = []
        job_raw = job_description
        job_processed = self.preprocess_text(job_description)
        job_keywords = self.extract_keywords(job_description)

        resume_texts = []
        filtered_resumes = []

        for resume_file in resume_files:
            text = self.extract_text_from_pdf(resume_file['path'])
            if not text.strip():
                print(f"Skipping {resume_file['filename']} due to empty content")
                continue
            processed_text = self.preprocess_text(text)
            filtered_resumes.append({
                'filename': resume_file['filename'],
                'original_text': text,
                'processed_text': processed_text
            })
            resume_texts.append(text)

        self.resumes = filtered_resumes

        if not resume_texts:
            return []
       
        # FIXED: Fit TF-IDF vectorizer on entire corpus (job description + all resumes) at once
        all_documents = [job_raw] + resume_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_documents)
        
        # Calculate similarities between job description (index 0) and each resume (indices 1+)
        job_vector = tfidf_matrix[0:1]  # Job description vector
        resume_vectors = tfidf_matrix[1:]  # All resume vectors
        similarities = cosine_similarity(job_vector, resume_vectors).flatten()

        for i, resume in enumerate(self.resumes):
            if i >= len(similarities):
                print(f"Warning: Missing similarity score for {resume['filename']}")
                continue

            tfidf_score = similarities[i] * 100
            keyword_score = self.calculate_keyword_match_score(resume['original_text'], job_keywords)
            experience_score = self.calculate_experience_score(resume['original_text'])
            education_score = self.calculate_education_score(resume['original_text'])
            skills_score = self.calculate_skills_score(resume['original_text'], job_description)

            final_score = (
             tfidf_score * 0.15 +         # reduced to avoid bias from irrelevant text matches
             keyword_score * 0.20 +       # increased for job-specific relevance
             experience_score * 0.20 +    # fair weight for actual hands-on experience
             education_score * 0.15 +     # education supports but doesn't override experience
             skills_score * 0.30     # slightly more important in skill-based roles
             )


            self.scores.append({
                'filename': resume['filename'],
                'final_score': round(final_score, 2),
                'tfidf_score': round(tfidf_score, 2),
                'keyword_score': round(keyword_score, 2),
                'experience_score': round(experience_score, 2),
                'education_score': round(education_score, 2),
                'skills_score': round(skills_score, 2),
                'matched_keywords': list(set(resume['original_text'].lower().split()) & set([kw.lower() for kw in job_keywords]))[:10]
            })

        self.scores.sort(key=lambda x: x['final_score'], reverse=True)
        return self.scores

ranker = ResumeRanker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    job_description = request.form.get('job_description', '')
    files = request.files.getlist('resumes')
    if not job_description or not files:
        flash('Job description and resumes are required')
        return redirect(url_for('index'))

    resume_files = []
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            resume_files.append({'filename': filename, 'path': filepath})

    if not resume_files:
        flash('No valid resumes uploaded')
        return redirect(url_for('index'))

    results = ranker.rank_resumes(job_description, resume_files)
    return render_template('results.html', results=results, job_description=job_description)

@app.route('/download_report')
def download_report():
    if not ranker.scores:
        flash('No results available')
        return redirect(url_for('index'))

    df = pd.DataFrame(ranker.scores)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Rankings', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Rankings']
        format_header = workbook.add_format({'bold': True, 'text_wrap': True, 'fg_color': '#D7E4BC', 'border': 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, format_header)
        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, min(max(df[col].astype(str).str.len().max(), len(col)) + 2, 50))
    output.seek(0)
    filename = f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=filename)

@app.route('/api/rank', methods=['POST'])
def api_rank():
    data = request.get_json()
    job_description = data.get('job_description', '')
    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400
    return jsonify({'message': 'API available'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
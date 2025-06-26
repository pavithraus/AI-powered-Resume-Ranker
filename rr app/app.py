import os
import io
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import PyPDF2
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from fuzzywuzzy import fuzz
from dateutil.relativedelta import relativedelta
from dateutil import parser
import numpy as np

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
        self.job_description = ""
        self.resumes = []
        self.scores = []
        # Pre-fitted vectorizer for consistent scoring
        self.base_vectorizer = None
        self.base_vocabulary = None

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
        if not text or len(text.strip()) < 5:
            return ""
            
        if not nlp:
            # Enhanced preprocessing without spaCy
            text = text.lower()
            # Remove special characters but keep spaces and alphanumeric
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove very short words (less than 2 characters) and numbers
            words = [word for word in text.split() 
                    if len(word) > 2 and not word.isdigit()]
            return ' '.join(words)

        # Enhanced preprocessing with spaCy
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        try:
            doc = nlp(text.lower())
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop 
                     and not token.is_punct 
                     and token.is_alpha 
                     and len(token.text) > 2
                     and not token.like_num]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in spaCy processing: {e}")
            # Fallback to simple preprocessing
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            words = [word for word in text.split() 
                    if len(word) > 2 and not word.isdigit()]
            return ' '.join(words)

    def extract_keywords(self, text, top_n=30):
        if not nlp:
            # Simple keyword extraction without spaCy
            words = text.lower().split()
            # Filter out common stop words and short words
            filtered_words = [word for word in words 
                            if word not in STOP_WORDS 
                            and len(word) > 2]
            word_freq = Counter(filtered_words)
            return [word for word, _ in word_freq.most_common(top_n)]

        # Enhanced keyword extraction with spaCy
        doc = nlp(text)
        keywords = []
        
        # Extract named entities (organizations, products, etc.)
        keywords += [ent.text.lower() for ent in doc.ents 
                    if ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'GPE']]
        
        # Extract important nouns and adjectives
        keywords += [token.lemma_.lower() for token in doc 
                    if token.pos_ in ['NOUN', 'ADJ', 'PROPN'] 
                    and not token.is_stop 
                    and len(token.text) > 2
                    and token.is_alpha]
        
        keyword_freq = Counter(keywords)
        return [word for word, _ in keyword_freq.most_common(top_n)]

    def calculate_keyword_match_score(self, resume_text, job_keywords):
        resume_words = re.findall(r'\b\w+\b', resume_text.lower())
        resume_words = set(word for word in resume_words 
                          if word not in STOP_WORDS and len(word) > 2)
        job_words = set(word.lower() for word in job_keywords 
                       if word.lower() not in STOP_WORDS)
        
        if not job_words:
            return 0
            
        # Exact matches
        exact_matches = len(resume_words.intersection(job_words))
        
        # Fuzzy matches for partial matching
        fuzzy_matches = 0
        for job_word in job_words:
            if job_word not in resume_words:
                for resume_word in resume_words:
                    if fuzz.ratio(job_word, resume_word) > 85:
                        fuzzy_matches += 0.5
                        break
        
        total_matches = exact_matches + fuzzy_matches
        return min((total_matches / len(job_words)) * 100, 100)

    def calculate_experience_score(self, resume_text):
        resume_text = resume_text.lower()
        
        # Enhanced date pattern matching
        date_patterns = [
            r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s?\d{2,4})\s*(?:to|\u2013|-|–)\s*(present|current|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s?\d{2,4})',
            r'(\d{1,2}\/\d{2,4})\s*(?:to|\u2013|-|–)\s*(present|current|\d{1,2}\/\d{2,4})',
            r'(\d{4})\s*(?:to|\u2013|-|–)\s*(present|current|\d{4})'
        ]
        
        total_years = 0
        found_dates = set()
        
        for pattern in date_patterns:
            date_ranges = re.findall(pattern, resume_text, flags=re.IGNORECASE)
            
            for start_str, end_str in date_ranges:
                # Avoid counting the same date range multiple times
                date_key = f"{start_str}_{end_str}"
                if date_key in found_dates:
                    continue
                found_dates.add(date_key)
                
                try:
                    start_date = parser.parse(start_str, fuzzy=True, default=datetime(2000, 1, 1))
                    if 'present' in end_str.lower() or 'current' in end_str.lower():
                        end_date = datetime.now()
                    else:
                        end_date = parser.parse(end_str, fuzzy=True, default=datetime(2000, 1, 1))

                    if end_date < start_date:
                        continue

                    diff = relativedelta(end_date, start_date)
                    years = diff.years + (diff.months / 12)
                    if years > 0 and years < 50:  # Sanity check
                        total_years += years
                except Exception:
                    continue

        # Also look for explicit year mentions
        year_mentions = re.findall(r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', resume_text)
        for year_str in year_mentions:
            try:
                years = int(year_str)
                if 0 < years < 50:
                    total_years = max(total_years, years)
            except ValueError:
                continue

        total_years = max(total_years, 0)
        return min(int(total_years * 8), 100)  # Adjusted scaling

    def calculate_education_score(self, resume_text):
        resume_lower = resume_text.lower()
        
        education_levels = {
            'phd': 25, 'doctorate': 25, 'doctoral': 25,
            'master': 20, 'msc': 20, 'mba': 20, 'ms': 15,
            'bachelor': 15, 'bsc': 15, 'bs': 10, 'be': 10, 'btech': 15,
            'degree': 10, 'graduate': 8, 'diploma': 5,
            'certification': 5, 'certified': 3, 'certificate': 5
        }
        
        institutions = {
            'university': 5, 'college': 5, 'institute': 5, 'school': 3
        }
        
        score = 0
        for term, points in education_levels.items():
            if term in resume_lower:
                score += points
                
        for term, points in institutions.items():
            if term in resume_lower:
                score += points
                
        return min(score, 100)

    def calculate_skills_score(self, resume_text, job_description, tech_skills=None):
        if tech_skills is None:
            tech_skills = [
                # Programming Languages
                'python', 'r', 'sql', 'java', 'c++', 'c', 'javascript', 'scala', 'julia',
                
                # AI/ML Technologies
                'machine learning', 'ml', 'ai', 'artificial intelligence',
                'deep learning', 'nlp', 'natural language processing', 'computer vision',
                'neural networks', 'reinforcement learning',
                
                # ML Libraries/Frameworks
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
                'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy',
                'matplotlib', 'seaborn', 'plotly', 'opencv', 'transformers',
                
                # Data Tools
                'excel', 'tableau', 'power bi', 'qlik', 'looker',
                'data visualization', 'data analysis', 'statistics', 'statistical analysis',
                'data mining', 'data science', 'analytics', 'reporting',
                
                # Cloud & Infrastructure
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                'spark', 'hadoop', 'airflow', 'kafka', 'mlflow', 'databricks',
                
                # Databases
                'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch',
                
                # Development Tools
                'git', 'github', 'vscode', 'jupyter', 'colab', 'linux',
                
                # Methodologies
                'agile', 'scrum', 'devops', 'ci/cd',
                
                # Soft Skills
                'teamwork', 'leadership', 'problem-solving', 'communication',
                'presentation', 'project management'
            ]

        job_lower = job_description.lower()
        resume_lower = resume_text.lower()

        # Extract skills mentioned in job description
        job_skills = []
        for skill in tech_skills:
            skill_lower = skill.lower()
            if skill_lower in job_lower:
                # Determine importance based on context
                skill_contexts = len(re.findall(rf'\b{re.escape(skill_lower)}\b', job_lower))
                importance = min(skill_contexts * 0.5 + 1.0, 2.0)
                job_skills.append((skill_lower, importance))
            elif fuzz.partial_ratio(skill_lower, job_lower) > 80:
                job_skills.append((skill_lower, 0.7))

        if not job_skills:
            return 50  # Default score if no skills identified

        total_weight = 0
        matched_weight = 0

        for skill, weight in job_skills:
            total_weight += weight
            # Check for exact or fuzzy match in resume
            if skill in resume_lower:
                matched_weight += weight
            elif fuzz.partial_ratio(skill, resume_lower) > 80:
                matched_weight += weight * 0.8

        return min((matched_weight / total_weight) * 100, 100) if total_weight > 0 else 0

    def calculate_consistent_tfidf_score(self, job_processed, resume_processed):
        """
        Calculate TF-IDF score consistently regardless of number of resumes
        """
        # Check if texts are empty or too short
        if len(job_processed.strip()) < 10 or len(resume_processed.strip()) < 10:
            return 0
        
        # Create a more lenient vectorizer for better vocabulary matching
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Reduced to 1-2 grams for better matching
            min_df=1,            # Must appear at least once
            max_df=1.0,          # Can appear in all documents
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
        )
        
        try:
            # Always fit on both job and resume to ensure consistent vocabulary
            corpus = [job_processed, resume_processed]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Debug: Check if we have any features
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                print("Warning: No features extracted from texts")
                return 0
            
            # Calculate similarity
            job_vector = tfidf_matrix[0:1]
            resume_vector = tfidf_matrix[1:2]
            similarity = cosine_similarity(job_vector, resume_vector)[0][0]
            
            # If similarity is NaN or negative, return 0
            if np.isnan(similarity) or similarity < 0:
                return 0
                
            return min(similarity * 100, 100)  # Cap at 100%
            
        except Exception as e:
            print(f"Error in TF-IDF calculation: {e}")
            # Fallback to simple word overlap similarity
            return self.calculate_simple_text_similarity(job_processed, resume_processed)
    
    def calculate_simple_text_similarity(self, job_text, resume_text):
        """
        Fallback similarity calculation using simple word overlap
        """
        job_words = set(job_text.lower().split())
        resume_words = set(resume_text.lower().split())
        
        if not job_words or not resume_words:
            return 0
            
        intersection = job_words.intersection(resume_words)
        union = job_words.union(resume_words)
        
        if not union:
            return 0
            
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        return jaccard * 100

    def rank_resumes(self, job_description, resume_files):
        self.job_description = job_description
        self.resumes = []
        self.scores = []

        job_processed = self.preprocess_text(job_description)
        job_keywords = self.extract_keywords(job_description)

        processed_resumes = []

        # Process all resumes
        for resume_file in resume_files:
            text = self.extract_text_from_pdf(resume_file['path'])
            if not text.strip():
                print(f"Skipping {resume_file['filename']} due to empty content")
                continue
            
            resume_processed = self.preprocess_text(text)
            processed_resumes.append({
                'filename': resume_file['filename'],
                'original_text': text,
                'processed_text': resume_processed
            })

        if not processed_resumes:
            return []

        # Calculate scores for each resume individually for consistency
        for resume in processed_resumes:
            # Debug: Print processed texts length
            print(f"Processing {resume['filename']}")
            print(f"Job processed length: {len(job_processed)}")
            print(f"Resume processed length: {len(resume['processed_text'])}")
            
            # Calculate TF-IDF score consistently
            tfidf_score = self.calculate_consistent_tfidf_score(
                job_processed, 
                resume['processed_text']
            )
            
            print(f"TF-IDF Score for {resume['filename']}: {tfidf_score}")
            
            # Calculate other scores
            keyword_score = self.calculate_keyword_match_score(
                resume['original_text'], 
                job_keywords
            )
            experience_score = self.calculate_experience_score(resume['original_text'])
            education_score = self.calculate_education_score(resume['original_text'])
            skills_score = self.calculate_skills_score(
                resume['original_text'], 
                job_description
            )

            # Calculate final weighted score
            final_score = (
                tfidf_score * 0.25 +      # Increased weight for content similarity
                keyword_score * 0.20 +    # Keyword matching
                experience_score * 0.20 + # Experience
                education_score * 0.15 +  # Education
                skills_score * 0.20       # Skills matching
            )

            # Get matched keywords for display
            resume_words = set(re.findall(r'\b\w+\b', resume['original_text'].lower()))
            job_words = set(word.lower() for word in job_keywords)
            matched_keywords = list(resume_words.intersection(job_words))[:10]

            self.scores.append({
                'filename': resume['filename'],
                'final_score': round(final_score, 2),
                'tfidf_score': round(tfidf_score, 2),
                'keyword_score': round(keyword_score, 2),
                'experience_score': round(experience_score, 2),
                'education_score': round(education_score, 2),
                'skills_score': round(skills_score, 2),
                'matched_keywords': matched_keywords
            })

        # Sort by final score
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
        
        # Format headers
        format_header = workbook.add_format({
            'bold': True, 
            'text_wrap': True, 
            'fg_color': '#D7E4BC', 
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, format_header)
            
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).str.len().max(),
                len(col)
            ) + 2
            worksheet.set_column(i, i, min(max_length, 50))
            
    output.seek(0)
    filename = f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output, 
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
        as_attachment=True, 
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
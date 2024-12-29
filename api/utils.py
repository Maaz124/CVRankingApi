
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class ResumeRanker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResumeRanker, cls).__new__(cls)
            # Use a smaller model
            cls._instance.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            cls._instance.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        return cls._instance

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    def compute_similarity(self, emb1, emb2):
        return float(torch.nn.functional.cosine_similarity(emb1, emb2).item())

    def normalize_text(self, data_list):
        """Convert list of items to normalized text."""
        return " ".join(str(item) for item in data_list)

    def match_resume_to_job(self, resume_text, job_description):
        """Calculate similarity between resume text and job description."""
        try:
            emb1 = self.get_embedding(resume_text)
            emb2 = self.get_embedding(job_description)
            return self.compute_similarity(emb1, emb2)
        except Exception as e:
            print(f"Error in match_resume_to_job: {str(e)}")
            return 0.0

    def compute_skill_match(self, skills, job_description):
        """Calculate percentage of matching skills."""
        try:
            matched_skills = [
                skill for skill in skills 
                if str(skill).lower() in job_description.lower()
            ]
            return len(matched_skills) / len(skills) if skills else 0.0
        except Exception as e:
            print(f"Error in compute_skill_match: {str(e)}")
            return 0.0

    def rank_candidates(self, candidates, job_description):
        """Rank candidates based on their match with job description."""
        try:
            scores = []
            
            for candidate in candidates:
                # Normalize texts
                skills_text = self.normalize_text(candidate.skills)
                experience_text = self.normalize_text(candidate.experience)
                education_text = self.normalize_text(candidate.education)

                # Calculate scores
                skill_score = self.match_resume_to_job(skills_text, job_description)
                experience_score = self.match_resume_to_job(experience_text, job_description)
                education_score = self.match_resume_to_job(education_text, job_description)
                keyword_match_score = self.compute_skill_match(candidate.skills, job_description)

                # Calculate weighted total score
                total_score = (
                    0.5 * skill_score + 
                    0.4 * experience_score + 
                    0.1 * education_score + 
                    0.1 * keyword_match_score
                )

                scores.append({
                    "name": candidate.name,
                    "skill_score": round(skill_score, 3),
                    "experience_score": round(experience_score, 3),
                    "education_score": round(education_score, 3),
                    "keyword_match_score": round(keyword_match_score, 3),
                    "total_score": round(total_score, 3)
                })

            return sorted(scores, key=lambda x: x["total_score"], reverse=True)
        except Exception as e:
            print(f"Error in rank_candidates: {str(e)}")
            return []
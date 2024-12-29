from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class ResumeRanker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResumeRanker, cls).__new__(cls)
            
            # Set device
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {cls._instance.device}")
            
            # Load model and move to GPU
            cls._instance.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            cls._instance.model = cls._instance.model.to(cls._instance.device)
            
        return cls._instance

    def normalize_text(self, data_list):
        """Convert list of items to normalized text."""
        return " ".join(str(item) for item in data_list)

    def match_resume_to_job(self, resume_text, job_description):
        """Calculate similarity between resume text and job description."""
        try:
            # Encode texts and move tensors to GPU
            embeddings = self.model.encode(
                [resume_text, job_description],
                convert_to_tensor=True,
                device=self.device
            )
            
            # Calculate similarity on GPU
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            
            # Move result back to CPU and convert to Python float
            return float(similarity.cpu().item())
            
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

    @torch.no_grad()  # Disable gradient computation for inference
    def rank_candidates(self, candidates, job_description):
        """Rank candidates based on their match with job description."""
        try:
            scores = []
            
            # Process candidates in batches to maximize GPU utilization
            batch_size = 8  # Adjust based on your GPU memory
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_scores = []
                
                for candidate in batch:
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

                    batch_scores.append({
                        "name": candidate.name,
                        "skill_score": round(skill_score, 3),
                        "experience_score": round(experience_score, 3),
                        "education_score": round(education_score, 3),
                        "keyword_match_score": round(keyword_match_score, 3),
                        "total_score": round(total_score, 3)
                    })
                
                scores.extend(batch_scores)

            # Sort all scores
            return sorted(scores, key=lambda x: x["total_score"], reverse=True)
            
        except Exception as e:
            print(f"Error in rank_candidates: {str(e)}")
            return []

    def __del__(self):
        """Cleanup GPU memory when the instance is destroyed."""
        try:
            if hasattr(self, 'model'):
                del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass
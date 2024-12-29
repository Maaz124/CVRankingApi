from django.db import models

# Create your models here.
from django.db import models

class JobPosting(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']

class Candidate(models.Model):
    name = models.CharField(max_length=100)
    education = models.JSONField(help_text="List of education entries")
    skills = models.JSONField(help_text="List of skills")
    experience = models.JSONField(help_text="List of experience entries")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['-created_at']
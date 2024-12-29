
from rest_framework import serializers
from .models import JobPosting, Candidate

class JobPostingSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobPosting
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class CandidateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candidate
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class RankingResultSerializer(serializers.Serializer):
    name = serializers.CharField()
    total_score = serializers.FloatField()
    skill_score = serializers.FloatField()
    experience_score = serializers.FloatField()
    education_score = serializers.FloatField()
    keyword_match_score = serializers.FloatField()
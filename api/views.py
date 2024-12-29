from django.shortcuts import render

# Create your views here.

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import JobPosting, Candidate
from .serializers import (
    JobPostingSerializer, 
    CandidateSerializer,
    RankingResultSerializer
)
from .utils import ResumeRanker

class JobPostingViewSet(viewsets.ModelViewSet):
    queryset = JobPosting.objects.all()
    serializer_class = JobPostingSerializer

    @action(detail=True, methods=['post'])
    def rank_candidates(self, request, pk=None):
        """
        Rank all candidates for a specific job posting.
        """
        try:
            job_posting = self.get_object()
            candidates = Candidate.objects.all()
            
            if not candidates.exists():
                return Response(
                    {"error": "No candidates found in the system"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            ranker = ResumeRanker()
            rankings = ranker.rank_candidates(candidates, job_posting.description)
            
            serializer = RankingResultSerializer(rankings, many=True)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CandidateViewSet(viewsets.ModelViewSet):
    queryset = Candidate.objects.all()
    serializer_class = CandidateSerializer
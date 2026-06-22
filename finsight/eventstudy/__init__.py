"""Event-study (cumulative abnormal return) analysis around news."""

from finsight.eventstudy.study import EventStudyResult, event_study

__all__ = ["event_study", "EventStudyResult"]

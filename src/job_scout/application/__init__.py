"""Local, review-gated application preparation.

This package deliberately has no submit operation. It can inspect and fill
safe fields in a visible browser, then stops for the human's final review.
"""

from job_scout.application.ats import (
    ApplicantFacts,
    ATSName,
    FieldProposal,
    FormField,
    FormInspection,
    detect_ats,
)

__all__ = ["ApplicantFacts", "ATSName", "FieldProposal", "FormField", "FormInspection", "detect_ats"]

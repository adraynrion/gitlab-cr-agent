"""
GitLab webhook handlers for merge request events
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
import hmac
import hashlib

from src.models.gitlab_models import GitLabWebhookPayload, MergeRequestEvent
from src.services.gitlab_service import GitLabService
from src.services.review_service import ReviewService
from src.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

def verify_gitlab_token(request: Request) -> bool:
    """Verify GitLab webhook secret token"""
    gitlab_token = request.headers.get("X-Gitlab-Token", "")
    
    if not settings.gitlab_webhook_secret:
        return True  # Skip verification if no secret configured
    
    return hmac.compare_digest(gitlab_token, settings.gitlab_webhook_secret)

@router.post("/gitlab")
async def handle_gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle GitLab webhook events for merge requests
    
    Triggers AI code review when the configured tag is added to a merge request
    """
    
    # Verify webhook authentication
    if not verify_gitlab_token(request):
        logger.warning("Invalid GitLab webhook token received")
        raise HTTPException(status_code=401, detail="Invalid webhook token")
    
    # Parse webhook payload
    try:
        payload = await request.json()
        webhook_event = GitLabWebhookPayload(**payload)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")
    
    # Handle merge request events
    if webhook_event.object_kind != "merge_request":
        return {"status": "ignored", "reason": "Not a merge request event"}
    
    mr_event = MergeRequestEvent(**payload)
    
    # Check if the trigger tag is present
    trigger_tag = settings.gitlab_trigger_tag
    label_titles = [label.title for label in mr_event.object_attributes.labels]
    if trigger_tag not in label_titles:
        return {
            "status": "ignored",
            "reason": f"Trigger tag '{trigger_tag}' not found"
        }
    
    # Check if this is a relevant action
    relevant_actions = ["open", "update", "reopen"]
    if mr_event.object_attributes.action not in relevant_actions:
        return {
            "status": "ignored",
            "reason": f"Action '{mr_event.object_attributes.action}' not relevant"
        }
    
    # Queue background task for code review
    logger.info(f"Queueing review for MR {mr_event.object_attributes.iid}")
    
    background_tasks.add_task(
        process_merge_request_review,
        mr_event
    )
    
    return {
        "status": "processing",
        "merge_request_iid": mr_event.object_attributes.iid,
        "project_id": mr_event.project.id
    }

async def process_merge_request_review(mr_event: MergeRequestEvent):
    """
    Background task to process merge request review
    """
    try:
        gitlab_service = GitLabService()
        review_service = ReviewService()
        
        # Fetch merge request details and diff
        mr_details = await gitlab_service.get_merge_request(
            project_id=mr_event.project.id,
            mr_iid=mr_event.object_attributes.iid
        )
        
        mr_diff = await gitlab_service.get_merge_request_diff(
            project_id=mr_event.project.id,
            mr_iid=mr_event.object_attributes.iid
        )
        
        # Perform AI code review
        review_result = await review_service.review_merge_request(
            mr_details=mr_details,
            mr_diff=mr_diff,
            mr_event=mr_event
        )
        
        # Post review comment to GitLab
        comment = review_service.format_review_comment(review_result)
        
        await gitlab_service.post_merge_request_comment(
            project_id=mr_event.project.id,
            mr_iid=mr_event.object_attributes.iid,
            comment=comment
        )
        
        logger.info(f"Successfully posted review for MR {mr_event.object_attributes.iid}")
        
    except Exception as e:
        logger.error(f"Failed to process MR review: {e}")
        
        # Post error comment to GitLab
        try:
            error_comment = f"❌ **AI Code Review Failed**\n\nAn error occurred during the review process. Please check the logs for details."
            
            await gitlab_service.post_merge_request_comment(
                project_id=mr_event.project.id,
                mr_iid=mr_event.object_attributes.iid,
                comment=error_comment
            )
        except:
            pass  # Fail silently if we can't post the error comment
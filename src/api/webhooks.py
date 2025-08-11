"""
GitLab webhook handlers for merge request events
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from typing import Optional
import logging
import hmac
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.gitlab_models import GitLabWebhookPayload, MergeRequestEvent
from src.services.gitlab_service import GitLabService
from src.services.review_service import ReviewService
from src.config.settings import settings
from src.agents.code_reviewer import CodeReviewAgent

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiter for webhooks
limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)


def verify_gitlab_token(request: Request):
    """Verify GitLab webhook secret token"""
    gitlab_token = request.headers.get("X-Gitlab-Token", "")

    if not settings.gitlab_webhook_secret:
        logger.warning("GitLab webhook secret not configured - accepting all webhooks")
        return True  # Skip verification if no secret configured

    if not gitlab_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing webhook token",
                "message": "X-Gitlab-Token header is required",
                "type": "authentication_error",
            },
        )

    is_valid = hmac.compare_digest(gitlab_token, settings.gitlab_webhook_secret)

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid webhook token",
                "message": "Invalid X-Gitlab-Token provided",
                "type": "authentication_error",
            },
        )

    return True


@router.post("/gitlab")
@limiter.limit(settings.webhook_rate_limit)
async def handle_gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle GitLab webhook events for merge requests

    Triggers AI code review when the configured tag is added to a merge request
    """

    # Verify webhook authentication
    # Since verify_gitlab_token now raises HTTPException directly, we don't need try/catch
    verify_gitlab_token(request)

    # Parse webhook payload
    try:
        payload = await request.json()
        webhook_event = GitLabWebhookPayload(**payload)
    except ValueError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Webhook payload validation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid webhook payload",
                "message": "Webhook payload validation failed",
                "type": "validation_error",
            },
        )

    # Handle merge request events
    if webhook_event.object_kind != "merge_request":
        return {"status": "ignored", "reason": "Not a merge request event"}

    mr_event = MergeRequestEvent(**payload)

    # Check if the trigger tag is present
    trigger_tag = settings.gitlab_trigger_tag
    label_titles = [label.title for label in mr_event.object_attributes.labels]
    if trigger_tag not in label_titles:
        return {"status": "ignored", "reason": f"Trigger tag '{trigger_tag}' not found"}

    # Check if this is a relevant action
    relevant_actions = ["open", "update", "reopen"]
    if mr_event.object_attributes.action not in relevant_actions:
        return {
            "status": "ignored",
            "reason": f"Action '{mr_event.object_attributes.action}' not relevant",
        }

    # Queue background task for code review
    logger.info(f"Queueing review for MR {mr_event.object_attributes.iid}")

    background_tasks.add_task(
        process_merge_request_review,
        mr_event,
    )

    return {
        "status": "processing",
        "merge_request_iid": mr_event.object_attributes.iid,
        "project_id": mr_event.project.id,
    }


async def process_merge_request_review(
    mr_event: MergeRequestEvent, review_agent: Optional[CodeReviewAgent] = None
):
    """
    Background task to process merge request review
    """
    try:
        # Initialize services - create fresh instances for background task
        gitlab_service = GitLabService()

        # Initialize review agent if not provided
        if review_agent is None:
            from src.agents.code_reviewer import initialize_review_agent

            review_agent = await initialize_review_agent()

        review_service = ReviewService(review_agent=review_agent)

        # Fetch merge request details and diff
        mr_details = await gitlab_service.get_merge_request(
            project_id=mr_event.project.id, mr_iid=mr_event.object_attributes.iid
        )

        mr_diff = await gitlab_service.get_merge_request_diff(
            project_id=mr_event.project.id, mr_iid=mr_event.object_attributes.iid
        )

        # Perform AI code review
        review_result = await review_service.review_merge_request(
            mr_details=mr_details, mr_diff=mr_diff, mr_event=mr_event
        )

        # Post review comment to GitLab
        comment = review_service.format_review_comment(review_result)

        await gitlab_service.post_merge_request_comment(
            project_id=mr_event.project.id,
            mr_iid=mr_event.object_attributes.iid,
            comment=comment,
        )

        logger.info(
            f"Successfully posted review for MR {mr_event.object_attributes.iid}"
        )

    except Exception as e:
        logger.error(f"Failed to process MR review: {e}")

        # Post error comment to GitLab
        try:
            error_comment = "❌ **AI Code Review Failed**\n\nAn error occurred during the review process. Please check the logs for details."

            gitlab_service = GitLabService()
            await gitlab_service.post_merge_request_comment(
                project_id=mr_event.project.id,
                mr_iid=mr_event.object_attributes.iid,
                comment=error_comment,
            )
        except Exception as comment_error:
            logger.error(
                f"Failed to post error comment to GitLab MR {mr_event.object_attributes.iid}: {comment_error}",
                extra={
                    "project_id": mr_event.project.id,
                    "mr_iid": mr_event.object_attributes.iid,
                    "original_error": str(e),
                    "comment_error": str(comment_error),
                },
            )

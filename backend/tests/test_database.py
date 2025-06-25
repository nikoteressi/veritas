"""
Test database operations.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import User, VerificationResult
from app.crud import UserCRUD, VerificationResultCRUD


@pytest.mark.asyncio
async def test_create_user(test_db: AsyncSession):
    """Test creating a new user."""
    nickname = "test_user"
    user = await UserCRUD.create_user(test_db, nickname)
    
    assert user.nickname == nickname
    assert user.total_posts_checked == 0
    assert user.true_count == 0
    assert user.false_count == 0
    assert user.warning_issued is False


@pytest.mark.asyncio
async def test_get_or_create_user_new(test_db: AsyncSession):
    """Test getting or creating a new user."""
    nickname = "new_user"
    user = await UserCRUD.get_or_create_user(test_db, nickname)
    
    assert user.nickname == nickname
    assert user.total_posts_checked == 0


@pytest.mark.asyncio
async def test_get_or_create_user_existing(test_db: AsyncSession):
    """Test getting an existing user."""
    nickname = "existing_user"
    
    # Create user first
    user1 = await UserCRUD.create_user(test_db, nickname)
    
    # Get the same user
    user2 = await UserCRUD.get_or_create_user(test_db, nickname)
    
    assert user1.id == user2.id
    assert user1.nickname == user2.nickname


@pytest.mark.asyncio
async def test_update_user_reputation_true(test_db: AsyncSession):
    """Test updating user reputation with true verdict."""
    nickname = "test_user"
    
    # Create user and update reputation
    user = await UserCRUD.update_user_reputation(test_db, nickname, "true")
    
    assert user.nickname == nickname
    assert user.true_count == 1
    assert user.total_posts_checked == 1
    assert user.false_count == 0


@pytest.mark.asyncio
async def test_update_user_reputation_false(test_db: AsyncSession):
    """Test updating user reputation with false verdict."""
    nickname = "test_user"
    
    # Update reputation multiple times to trigger warning
    for _ in range(4):  # Assuming warning threshold is 3
        user = await UserCRUD.update_user_reputation(test_db, nickname, "false")
    
    assert user.false_count == 4
    assert user.total_posts_checked == 4
    assert user.warning_issued is True


@pytest.mark.asyncio
async def test_create_verification_result(test_db: AsyncSession):
    """Test creating a verification result."""
    result = await VerificationResultCRUD.create_verification_result(
        db=test_db,
        user_nickname="test_user",
        image_hash="test_hash",
        extracted_text="Test text",
        user_prompt="Test prompt",
        primary_topic="general",
        identified_claims='["Test claim"]',
        verdict="true",
        justification="Test justification",
        confidence_score=85,
        processing_time_seconds=10,
        model_used="test_model",
        tools_used='["test_tool"]'
    )
    
    assert result.user_nickname == "test_user"
    assert result.verdict == "true"
    assert result.confidence_score == 85


@pytest.mark.asyncio
async def test_get_verification_results_by_user(test_db: AsyncSession):
    """Test getting verification results for a user."""
    nickname = "test_user"
    
    # Create a verification result
    await VerificationResultCRUD.create_verification_result(
        db=test_db,
        user_nickname=nickname,
        image_hash="test_hash",
        extracted_text="Test text",
        user_prompt="Test prompt",
        primary_topic="general",
        identified_claims='["Test claim"]',
        verdict="true",
        justification="Test justification",
        confidence_score=85,
        processing_time_seconds=10,
        model_used="test_model",
        tools_used='["test_tool"]'
    )
    
    # Get results
    results = await VerificationResultCRUD.get_verification_results_by_user(
        test_db, nickname, limit=10
    )
    
    assert len(results) == 1
    assert results[0].user_nickname == nickname

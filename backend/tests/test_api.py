"""
Test API endpoints.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test the root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Veritas API is running"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "veritas-api"


@pytest.mark.asyncio
async def test_verify_post_no_file(client: AsyncClient):
    """Test verification endpoint without file."""
    response = await client.post("/api/v1/verify-post", data={"prompt": "test"})
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_verify_post_invalid_file(client: AsyncClient):
    """Test verification endpoint with invalid file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    data = {"prompt": "test prompt"}
    
    response = await client.post("/api/v1/verify-post", files=files, data=data)
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


@pytest.mark.asyncio
async def test_verify_post_valid_image(client: AsyncClient, sample_image_bytes):
    """Test verification endpoint with valid image."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"prompt": "Is this accurate?"}
    
    response = await client.post("/api/v1/verify-post", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert "status" in result
    # Note: This will likely fail until the full agent is implemented
    # but it tests the basic endpoint structure


@pytest.mark.asyncio
async def test_get_user_reputation_new_user(client: AsyncClient):
    """Test getting reputation for a new user."""
    response = await client.get("/api/v1/user-reputation/new_user")
    assert response.status_code == 200
    
    data = response.json()
    assert data["nickname"] == "new_user"
    assert data["total_posts_checked"] == 0
    assert data["true_count"] == 0
    assert data["false_count"] == 0


@pytest.mark.asyncio
async def test_reputation_stats(client: AsyncClient):
    """Test getting reputation statistics."""
    response = await client.get("/api/v1/reputation-stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_users" in data
    assert "total_posts_checked" in data
    assert "accuracy_rate" in data


@pytest.mark.asyncio
async def test_users_with_warnings(client: AsyncClient):
    """Test getting users with warnings."""
    response = await client.get("/api/v1/users-with-warnings")
    assert response.status_code == 200
    
    data = response.json()
    assert "users" in data
    assert "total_count" in data
    assert isinstance(data["users"], list)


@pytest.mark.asyncio
async def test_reputation_leaderboard(client: AsyncClient):
    """Test getting reputation leaderboard."""
    response = await client.get("/api/v1/leaderboard")
    assert response.status_code == 200
    
    data = response.json()
    assert "leaderboard" in data
    assert "total_users" in data
    assert isinstance(data["leaderboard"], list)

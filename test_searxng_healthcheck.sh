#!/bin/bash

# Test script for SearXNG healthcheck configuration
# This script validates that the wget-based healthcheck is working correctly

echo "🔍 Testing SearXNG Healthcheck Configuration"
echo "============================================="

# Check if SearXNG container is running
echo "1. Checking if SearXNG container is running..."
if docker ps --filter "name=veritas-searxng" --format "{{.Names}}" | grep -q "veritas-searxng"; then
    echo "✅ SearXNG container is running"
else
    echo "❌ SearXNG container is not running"
    exit 1
fi

# Check container health status
echo ""
echo "2. Checking container health status..."
HEALTH_STATUS=$(docker inspect veritas-searxng --format='{{.State.Health.Status}}')
echo "Health Status: $HEALTH_STATUS"

if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo "✅ Container is healthy"
elif [ "$HEALTH_STATUS" = "starting" ]; then
    echo "⏳ Container health is starting (this is normal for new containers)"
else
    echo "❌ Container is not healthy: $HEALTH_STATUS"
fi

# Test the healthcheck command directly
echo ""
echo "3. Testing healthcheck command directly..."
if docker exec veritas-searxng wget --quiet --spider --tries=1 --timeout=5 http://localhost:8080/healthz; then
    echo "✅ Direct healthcheck command succeeded"
else
    echo "❌ Direct healthcheck command failed"
fi

# Test that wget is available
echo ""
echo "4. Verifying wget availability..."
if docker exec veritas-searxng which wget > /dev/null 2>&1; then
    echo "✅ wget is available in the container"
    WGET_VERSION=$(docker exec veritas-searxng wget --version | head -1)
    echo "   Version: $WGET_VERSION"
else
    echo "❌ wget is not available in the container"
fi

# Test that curl is NOT available (confirming the need for wget)
echo ""
echo "5. Confirming curl is not available..."
if docker exec veritas-searxng which curl > /dev/null 2>&1; then
    echo "⚠️  curl is available in the container (wget replacement was still beneficial)"
else
    echo "✅ curl is not available in the container (confirming need for wget)"
fi

# Show recent health check logs
echo ""
echo "6. Recent health check logs:"
docker inspect veritas-searxng --format='{{range .State.Health.Log}}{{.Start}} - Exit Code: {{.ExitCode}}{{if .Output}} - Output: {{.Output}}{{end}}
{{end}}' | tail -5

echo ""
echo "🎉 SearXNG healthcheck test completed!"

#!/usr/bin/env python3
"""
Test script for Bearer token authentication
"""

import os
from typing import Optional

import requests

# Configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key-12345")


def test_endpoint(
    path: str,
    headers: Optional[dict] = None,
    expected_status: int = 200,
    test_name: str = "",
) -> None:
    """Test an endpoint with given headers and expected status"""
    url = f"{BASE_URL}{path}"
    headers = headers or {}

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == expected_status:
            print(f"✅ {test_name}: PASSED (Status: {response.status_code})")
        else:
            print(f"❌ {test_name}: FAILED")
            print(f"   Expected: {expected_status}, Got: {response.status_code}")
            print(f"   Response: {response.text}")

        return response
    except Exception as e:
        print(f"❌ {test_name}: ERROR - {str(e)}")
        return None


def main():
    print("=" * 60)
    print("Bearer Token Authentication Test Suite")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    print()

    # Test 1: Root endpoint should always work without auth
    print("1. Testing public endpoints:")
    test_endpoint("/", test_name="Root endpoint (always public)")
    print()

    # Test 2: Protected endpoints without auth should fail (when API_KEY is set)
    if API_KEY:
        print("2. Testing protected endpoints without authentication:")
        test_endpoint(
            "/health/live",
            expected_status=401,
            test_name="Health live endpoint without auth",
        )
        test_endpoint(
            "/health/ready",
            expected_status=401,
            test_name="Health ready endpoint without auth",
        )
        test_endpoint(
            "/health/status",
            expected_status=401,
            test_name="Health status endpoint without auth",
        )
        test_endpoint(
            "/webhook/gitlab",
            expected_status=401,
            test_name="Webhook endpoint without auth",
        )
    else:
        print(
            "2. Skipping protected endpoint tests (API_KEY not set - all endpoints are public)"
        )
    print()

    # Test 3: Protected endpoints with invalid Bearer token
    if API_KEY:
        print("3. Testing protected endpoints with invalid Bearer token:")
        test_endpoint(
            "/health/status",
            headers={"Authorization": "Bearer invalid-token"},
            expected_status=401,
            test_name="Invalid Bearer token",
        )
        print()

        # Test 4: Protected endpoint with malformed auth header
        print("4. Testing protected endpoint with malformed auth header:")
        test_endpoint(
            "/health/status",
            headers={
                "Authorization": "Basic dGVzdDp0ZXN0"
            },  # Basic auth instead of Bearer
            expected_status=401,
            test_name="Wrong auth type (Basic instead of Bearer)",
        )
        print()

        # Test 5: All protected endpoints with valid Bearer token
        print("5. Testing all protected endpoints with valid Bearer token:")
        for endpoint in ["/health/live", "/health/ready", "/health/status"]:
            response = test_endpoint(
                endpoint,
                headers={"Authorization": f"Bearer {API_KEY}"},
                expected_status=200,
                test_name=f"Valid Bearer token - {endpoint}",
            )
        print()

        # Test 6: Verify WWW-Authenticate header in 401 responses
        print("6. Testing WWW-Authenticate header in 401 responses:")
        response = requests.get(f"{BASE_URL}/health/status")
        if response.status_code == 401:
            www_auth = response.headers.get("WWW-Authenticate")
            if www_auth == "Bearer":
                print("✅ WWW-Authenticate header present and correct")
            else:
                print(f"❌ WWW-Authenticate header incorrect: {www_auth}")
        print()
    else:
        print(
            "3-6. Skipping authentication tests (API_KEY not set - all endpoints are public)"
        )
        print()

        # Test that all endpoints work without auth when API_KEY is not set
        print("Testing all endpoints work without authentication:")
        for endpoint in ["/health/live", "/health/ready", "/health/status"]:
            test_endpoint(
                endpoint,
                expected_status=200,
                test_name=f"No auth required - {endpoint}",
            )
        print()

    print("=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

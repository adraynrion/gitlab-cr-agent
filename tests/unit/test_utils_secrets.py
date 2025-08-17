"""
Tests for src/utils/secrets.py
"""

import os
from unittest.mock import patch

import pytest

from src.utils.secrets import (
    SecretManager,
    clear_secrets_cache,
    get_secret_manager,
    get_secure_setting,
)


class TestSecretManager:
    """Test SecretManager class"""

    def test_secret_manager_creation(self):
        """Test SecretManager initialization"""
        manager = SecretManager()
        assert manager is not None
        assert hasattr(manager, "_encryption_key")
        assert hasattr(manager, "_cipher")
        assert hasattr(manager, "_secrets_cache")

    def test_secret_storage_and_retrieval(self):
        """Test storing and retrieving secrets"""
        manager = SecretManager()

        # Test secret retrieval with environment variable
        with patch.dict(os.environ, {"TEST_SECRET": "secret_value"}):
            secret = manager.get_secret("test", "TEST_SECRET", "default")
            assert secret == "secret_value"

    def test_secret_default_value(self):
        """Test secret retrieval with default value"""
        manager = SecretManager()

        # Test with non-existent environment variable
        secret = manager.get_secret("nonexistent", "NONEXISTENT_VAR", "default_value")
        assert secret == "default_value"

    def test_secret_caching(self):
        """Test secret caching mechanism"""
        manager = SecretManager()

        with patch.dict(os.environ, {"CACHED_SECRET": "cached_value"}):
            # First retrieval should cache the secret
            secret1 = manager.get_secret("cached", "CACHED_SECRET")
            # Second retrieval should use cache
            secret2 = manager.get_secret("cached", "CACHED_SECRET")
            assert secret1 == secret2 == "cached_value"

    def test_clear_individual_secret(self):
        """Test clearing individual secrets from cache"""
        manager = SecretManager()

        with patch.dict(os.environ, {"CLEAR_TEST": "test_value"}):
            # Store a secret
            manager.get_secret("clear_test", "CLEAR_TEST")
            # Clear it
            manager.clear_secret("clear_test")
            # Should still retrieve from environment
            secret = manager.get_secret("clear_test", "CLEAR_TEST")
            assert secret == "test_value"

    def test_clear_all_secrets(self):
        """Test clearing all secrets from cache"""
        manager = SecretManager()

        with patch.dict(os.environ, {"SECRET1": "value1", "SECRET2": "value2"}):
            # Store multiple secrets
            manager.get_secret("secret1", "SECRET1")
            manager.get_secret("secret2", "SECRET2")
            # Clear all
            manager.clear_all_secrets()
            # Should still retrieve from environment
            secret1 = manager.get_secret("secret1", "SECRET1")
            secret2 = manager.get_secret("secret2", "SECRET2")
            assert secret1 == "value1"
            assert secret2 == "value2"

    def test_secret_encryption_key_from_environment(self):
        """Test encryption key from environment variable"""
        import base64

        from cryptography.fernet import Fernet

        # Generate a valid key
        test_key = Fernet.generate_key()
        encoded_key = base64.urlsafe_b64encode(test_key).decode()

        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": encoded_key}):
            manager = SecretManager()
            assert manager._encryption_key == test_key

    def test_secret_encryption_error_handling(self):
        """Test encryption error handling"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": "invalid-key"}):
            # Should handle invalid key gracefully
            manager = SecretManager()
            assert manager is not None


class TestSecretManagerGlobal:
    """Test global secret manager functions"""

    def test_get_secret_manager_singleton(self):
        """Test secret manager singleton behavior"""
        manager1 = get_secret_manager()
        manager2 = get_secret_manager()
        assert manager1 is manager2

    def test_get_secure_setting(self):
        """Test get_secure_setting convenience function"""
        with patch.dict(os.environ, {"SECURE_TEST": "secure_value"}):
            setting = get_secure_setting("secure_test", "SECURE_TEST", "default")
            assert setting == "secure_value"

    def test_get_secure_setting_with_default(self):
        """Test get_secure_setting with default value"""
        setting = get_secure_setting("nonexistent", "NONEXISTENT_VAR", "default_value")
        assert setting == "default_value"

    def test_clear_secrets_cache_global(self):
        """Test global clear_secrets_cache function"""
        # This should not raise an error
        clear_secrets_cache()
        assert True

    def test_secret_manager_lifecycle(self):
        """Test complete secret manager lifecycle"""
        # Test manager creation
        manager1 = get_secret_manager()
        manager2 = get_secret_manager()
        assert manager1 is manager2  # Should be cached

        # Test secret retrieval with environment variable
        with patch.dict(os.environ, {"LIFECYCLE_SECRET": "lifecycle_value"}):
            secret = manager1.get_secret("test", "LIFECYCLE_SECRET", "default")
            assert secret == "lifecycle_value"

        # Test cache clearing
        clear_secrets_cache()
        assert True


class TestSecretManagerEdgeCases:
    """Test edge cases and error scenarios"""

    def test_secret_with_none_value(self):
        """Test secret retrieval when environment variable is None"""
        manager = SecretManager()

        # Mock os.getenv to return None
        with patch("os.getenv", return_value=None):
            secret = manager.get_secret("none_test", "NONE_VAR", "default")
            assert secret == "default"

    def test_secret_without_default(self):
        """Test secret retrieval without default value"""
        manager = SecretManager()

        with patch("os.getenv", return_value=None):
            secret = manager.get_secret("no_default", "NO_DEFAULT_VAR")
            assert secret is None

    def test_secret_manager_destructor(self):
        """Test secret manager destructor cleanup"""
        manager = SecretManager()

        # Store a secret
        with patch.dict(os.environ, {"DESTRUCTOR_TEST": "test_value"}):
            manager.get_secret("destructor", "DESTRUCTOR_TEST")

        # Test destructor (should not raise exception)
        try:
            manager.__del__()
        except Exception:
            pytest.fail("SecretManager destructor should not raise exceptions")

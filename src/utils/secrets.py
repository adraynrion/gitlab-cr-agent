"""
Secure secret management utilities for sensitive configuration data
"""

import base64
import logging
import os
from typing import Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Secure secret management with in-memory encryption and cleanup
    """

    def __init__(self):
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        self._secrets_cache = {}

    def _get_or_create_encryption_key(self) -> bytes:
        """Generate or retrieve encryption key for runtime secret protection"""
        # In production, this should come from a secure key management service
        key_env = os.getenv("SECRET_ENCRYPTION_KEY")
        if key_env:
            try:
                return base64.urlsafe_b64decode(key_env.encode())
            except Exception as e:
                logger.warning(f"Invalid encryption key in environment: {e}")

        # Generate a runtime key (will be different each run for security)
        # In production, you would use AWS KMS, HashiCorp Vault, etc.
        key: bytes = Fernet.generate_key()
        logger.info("Generated runtime encryption key for secret management")
        return key

    def get_secret(
        self, secret_name: str, env_var: str, default: Optional[str] = None
    ) -> Optional[str]:
        """
        Securely retrieve and cache a secret with in-memory encryption

        Args:
            secret_name: Internal name for the secret
            env_var: Environment variable name
            default: Default value if secret not found

        Returns:
            Decrypted secret value or default
        """
        # Check cache first
        if secret_name in self._secrets_cache:
            try:
                encrypted_value = self._secrets_cache[secret_name]
                decrypted_bytes: bytes = self._cipher.decrypt(encrypted_value)
                return decrypted_bytes.decode()
            except Exception as e:
                logger.warning(f"Failed to decrypt cached secret {secret_name}: {e}")
                # Remove corrupted cache entry
                del self._secrets_cache[secret_name]

        # Get from environment
        secret_value = os.getenv(env_var, default)
        if secret_value is None:
            return None

        # Encrypt and cache
        try:
            encrypted_value = self._cipher.encrypt(secret_value.encode())
            self._secrets_cache[secret_name] = encrypted_value

            # Clear the plain text value from memory
            secret_value = None

            # Return decrypted value
            decrypted_result: bytes = self._cipher.decrypt(encrypted_value)
            return decrypted_result.decode()
        except Exception as e:
            logger.error(f"Failed to encrypt secret {secret_name}: {e}")
            return secret_value

    def clear_secret(self, secret_name: str) -> None:
        """Remove a secret from cache"""
        if secret_name in self._secrets_cache:
            del self._secrets_cache[secret_name]
            logger.debug(f"Cleared secret {secret_name} from cache")

    def clear_all_secrets(self) -> None:
        """Clear all cached secrets"""
        self._secrets_cache.clear()
        logger.info("Cleared all secrets from cache")

    def __del__(self):
        """Cleanup secrets on object destruction"""
        try:
            self.clear_all_secrets()
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Global secret manager instance
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """Get or create the global secret manager instance"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


def get_secure_setting(
    secret_name: str, env_var: str, default: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to get a secure setting

    Args:
        secret_name: Internal name for the secret
        env_var: Environment variable name
        default: Default value if not found

    Returns:
        Secret value or default
    """
    return get_secret_manager().get_secret(secret_name, env_var, default)


def clear_secrets_cache() -> None:
    """Clear all cached secrets"""
    global _secret_manager
    if _secret_manager:
        _secret_manager.clear_all_secrets()

"""Password hashing with scrypt, from the standard library.

A stored hash names its own parameters, so they can be raised later without
invalidating existing passwords: an old hash still verifies with the values
written beside it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

# RFC 7914's interactive-login parameters: 16 MiB and roughly 50 ms per hash.
N, R, P = 2**14, 8, 1
KEY_BYTES = 64
# scrypt reads the whole password; a cap keeps one request from being costly.
MAX_PASSWORD_BYTES = 1024


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$key``, all that verifying needs."""
    salt = secrets.token_bytes(16)
    key = _derive(password, salt, N, R, P)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = _derive(password, base64.b64decode(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    secret = password.encode()
    if len(secret) > MAX_PASSWORD_BYTES:
        raise ValueError("password too long")
    # scrypt needs about 128 * n * r bytes; room to raise the parameters later.
    return hashlib.scrypt(secret, salt=salt, n=n, r=r, p=p, dklen=KEY_BYTES, maxmem=64 * 2**20)


__all__ = ["MAX_PASSWORD_BYTES", "hash_password", "verify_password"]

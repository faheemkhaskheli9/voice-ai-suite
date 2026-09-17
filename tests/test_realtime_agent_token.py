import pytest

from realtime_agent.token import create_access_token, decode_access_token


def test_token_roundtrips_with_grant():
    token = create_access_token(
        "api-key", "api-secret", identity="test-client", room="room-1"
    )
    claims = decode_access_token(token, "api-secret")
    assert claims["iss"] == "api-key"
    assert claims["sub"] == "test-client"
    assert claims["video"]["room"] == "room-1"
    assert claims["video"]["roomJoin"] is True
    assert claims["exp"] > claims["iat"]


def test_wrong_secret_fails_verification():
    token = create_access_token("k", "right", identity="c", room="r")
    import jwt

    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(token, "wrong")


def test_missing_secret_rejected():
    with pytest.raises(ValueError):
        create_access_token("k", "", identity="c", room="r")


def test_non_positive_ttl_rejected():
    with pytest.raises(ValueError):
        create_access_token("k", "s", identity="c", room="r", ttl_seconds=0)

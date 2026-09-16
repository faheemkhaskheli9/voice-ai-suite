"""Tests for issue #19: Django project skeleton with a dashboard shell."""

import pytest
from django.urls import reverse

from dashboard.features import FEATURES

pytestmark = pytest.mark.django_db


def test_unauthenticated_dashboard_redirects_to_login(client):
    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 302
    assert resp.url.startswith(reverse("login"))


def test_authenticated_dashboard_lists_all_four_features(client, django_user_model):
    user = django_user_model.objects.create_user(username="user1", password="pw12345")
    client.force_login(user)

    resp = client.get(reverse("dashboard"))

    assert resp.status_code == 200
    assert len(FEATURES) == 4
    for feature in FEATURES:
        assert feature.title.encode() in resp.content
        assert feature.description.encode() in resp.content


def test_each_feature_links_to_a_real_working_flow(client, django_user_model):
    user = django_user_model.objects.create_user(username="user2", password="pw12345")
    client.force_login(user)

    for feature in FEATURES:
        url = reverse("feature", kwargs={"slug": feature.slug})
        assert url.encode() in client.get(reverse("dashboard")).content
        resp = client.get(url, follow=True)
        assert resp.status_code == 200
        assert feature.title.encode() in resp.content


def test_unauthenticated_feature_page_also_redirects_to_login(client):
    resp = client.get(reverse("feature", kwargs={"slug": FEATURES[0].slug}))
    assert resp.status_code == 302
    assert resp.url.startswith(reverse("login"))


def test_unknown_feature_slug_is_a_404(client, django_user_model):
    user = django_user_model.objects.create_user(username="user3", password="pw12345")
    client.force_login(user)

    resp = client.get(reverse("feature", kwargs={"slug": "not-a-real-feature"}))
    assert resp.status_code == 404

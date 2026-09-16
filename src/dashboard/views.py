from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from .features import FEATURES, FEATURES_BY_SLUG


@login_required
def index(request: HttpRequest) -> HttpResponse:
    return render(request, "dashboard/index.html", {"features": FEATURES})


@login_required
def feature_stub(request: HttpRequest, slug: str) -> HttpResponse:
    """Landing page for a feature slug: forwards to the real feature app's
    own view once one exists (`Feature.url_name`), otherwise renders a
    placeholder (Phases 2-5 not built yet).
    """
    feature = FEATURES_BY_SLUG.get(slug)
    if feature is None:
        raise Http404(f"Unknown feature slug: {slug!r}")
    if feature.url_name:
        return redirect(feature.url_name)
    return render(request, "dashboard/feature_stub.html", {"feature": feature})

"""Feature implementations for reconstruction and display tabs.

The ``REC_FEATURES`` and ``DISP_FEATURES`` dicts are the in-process
registries the tabs consume. Mutate them through
:func:`register_rec_feature` / :func:`register_disp_feature` rather than
poking the dict directly, so name collisions raise rather than silently
shadow.

Entry-point groups:
  * ``cohere_ui.jupyter_gui.rec_features`` provides features for RecTab
  * ``cohere_ui.jupyter_gui.disp_features`` provides features for DispTab

Both are loaded lazily by :func:`get_rec_features` /
:func:`get_disp_features`. RecTab and DispTab call those, so any third
party that ships entry points under those groups will see their
features mounted automatically the first time the GUI is opened.
"""

from typing import Dict, Type

from cohere_ui.jupyter_gui.features.base import Feature
from cohere_ui.jupyter_gui.features.rec_features import (
    GAFeature, LowResolutionFeature, ShrinkWrapFeature,
    PhaseConstrainFeature, PCDIFeature, TwinFeature,
    AverageFeature, ProgressFeature, LiveFeature
)
from cohere_ui.jupyter_gui.features.disp_features import (
    CropFeature, InterpolationFeature, ResolutionFeature,
    ReciprocalFeature, StrainFeature, DisplacementFeature
)

REC_FEATURES: Dict[str, Type[Feature]] = {
    'GA': GAFeature,
    'low resolution': LowResolutionFeature,
    'shrink wrap': ShrinkWrapFeature,
    'phase constrain': PhaseConstrainFeature,
    'pcdi': PCDIFeature,
    'twin': TwinFeature,
    'average': AverageFeature,
    'progress': ProgressFeature,
    'live': LiveFeature,
}

DISP_FEATURES: Dict[str, Type[Feature]] = {
    'crop': CropFeature,
    'interpolation': InterpolationFeature,
    'resolution': ResolutionFeature,
    'reciprocal': ReciprocalFeature,
    'strain': StrainFeature,
    'displacement': DisplacementFeature,
}

_REC_ENTRY_POINT_GROUP = 'cohere_ui.jupyter_gui.rec_features'
_DISP_ENTRY_POINT_GROUP = 'cohere_ui.jupyter_gui.disp_features'
_REC_EP_LOADED = False
_DISP_EP_LOADED = False


def _register_into(registry: Dict[str, Type[Feature]], name: str,
                   feature_cls: Type[Feature]) -> None:
    """Shared validator for the two public register helpers."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('feature name must be a non-empty string')
    if not (isinstance(feature_cls, type) and issubclass(feature_cls, Feature)):
        raise TypeError(
            f'feature_cls must be a Feature subclass, got {feature_cls!r}'
        )
    if name in registry:
        raise ValueError(f'feature already registered: {name!r}')
    registry[name] = feature_cls


def register_rec_feature(name: str, feature_cls: Type[Feature]) -> None:
    """Register a reconstruction feature under ``name``.

    Affects subsequently-constructed ``RecTab`` instances. Re-registering
    the same name raises ``ValueError``. Call :func:`unregister_rec_feature`
    first if you intend to replace a built-in.
    """
    _register_into(REC_FEATURES, name, feature_cls)


def register_disp_feature(name: str, feature_cls: Type[Feature]) -> None:
    """Register a visualization (Postprocess) feature under ``name``.

    Same semantics as :func:`register_rec_feature` but mounted into
    ``DispTab`` instead of ``RecTab``.
    """
    _register_into(DISP_FEATURES, name, feature_cls)


def unregister_rec_feature(name: str) -> bool:
    """Remove a rec feature from the registry. Returns True if removed."""
    return REC_FEATURES.pop(name, None) is not None


def unregister_disp_feature(name: str) -> bool:
    """Remove a disp feature from the registry. Returns True if removed."""
    return DISP_FEATURES.pop(name, None) is not None


def _load_entry_points(group: str, registry: Dict[str, Type[Feature]]) -> None:
    """Load and register entry points from ``group`` into ``registry``."""
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        eps = entry_points(group=group)
    except TypeError:
        eps = entry_points().get(group, ())
    import logging
    log = logging.getLogger(__name__)
    for ep in eps:
        try:
            obj = ep.load()
            if not (isinstance(obj, type) and issubclass(obj, Feature)):
                log.warning(
                    'Feature entry point %s.%r resolved to %r, expected Feature subclass',
                    group, ep.name, type(obj).__name__,
                )
                continue
            try:
                _register_into(registry, ep.name, obj)
            except ValueError as e:
                log.warning('Feature entry point %s.%r skipped: %s',
                            group, ep.name, e)
        except Exception as e:
            log.warning('Feature entry point %s.%r failed to load: %s',
                        group, ep.name, e)


def get_rec_features() -> Dict[str, Type[Feature]]:
    """Return the current rec-feature registry, loading entry points once."""
    global _REC_EP_LOADED
    if not _REC_EP_LOADED:
        _REC_EP_LOADED = True
        _load_entry_points(_REC_ENTRY_POINT_GROUP, REC_FEATURES)
    return REC_FEATURES


def get_disp_features() -> Dict[str, Type[Feature]]:
    """Return the current disp-feature registry, loading entry points once."""
    global _DISP_EP_LOADED
    if not _DISP_EP_LOADED:
        _DISP_EP_LOADED = True
        _load_entry_points(_DISP_ENTRY_POINT_GROUP, DISP_FEATURES)
    return DISP_FEATURES


__all__ = [
    'Feature',
    'REC_FEATURES', 'DISP_FEATURES',
    'register_rec_feature', 'register_disp_feature',
    'unregister_rec_feature', 'unregister_disp_feature',
    'get_rec_features', 'get_disp_features',
    'GAFeature', 'LowResolutionFeature', 'ShrinkWrapFeature',
    'PhaseConstrainFeature', 'PCDIFeature', 'TwinFeature',
    'AverageFeature', 'ProgressFeature', 'LiveFeature',
    'CropFeature', 'InterpolationFeature', 'ResolutionFeature',
    'ReciprocalFeature', 'StrainFeature', 'DisplacementFeature',
]

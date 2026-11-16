"""Load independent MANO/FLAME application switches for the normal entry point."""
import json
from pathlib import Path

DEFAULT_SETTINGS = Path(__file__).resolve().parent/'settings.json'
MODEL_DEFAULTS = {'use_mano_hand_model': True, 'use_flame_face_model': True}


def resolve_model_settings(path=None, hand_model=None, face_model=None):
    source = Path(path).resolve() if path is not None else DEFAULT_SETTINGS
    try:
        data = json.loads(source.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read model settings {source}: {error}') from error
    if not isinstance(data, dict):
        raise ValueError('Model settings must be a JSON object')
    unknown = set(data)-set(MODEL_DEFAULTS)
    if unknown:
        raise ValueError('Unknown model settings: ' + ', '.join(sorted(unknown)))
    if any(type(value) is not bool for value in data.values()):
        raise ValueError('Model settings must use JSON true/false (no quotes)')
    configured = {**MODEL_DEFAULTS, **data}
    requested = configured.copy()
    for key, override in [('use_mano_hand_model', hand_model), ('use_flame_face_model', face_model)]:
        if override is not None:
            if type(override) is not bool:
                raise ValueError(f'{key} override must be a boolean')
            requested[key] = override
    return {'file': str(source), 'configured': configured, 'requested': requested}

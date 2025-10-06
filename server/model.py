from __future__ import annotations
from pydantic import BaseModel, Field, create_model
from typing import get_origin, TypeVar, Type, get_type_hints, Any, Mapping
from Agent.main import BuildState

def _default_for_type(tp):
    origin = get_origin(tp) or tp
    if origin in (list,):
        return Field(default_factory=list)
    if origin in (dict,):
        return Field(default_factory=dict)
    if origin is str:
        return ""
    if origin is int:
        return 0
    if origin is float:
        return 0.0
    if origin is bool:
        return False
    return None


class CompatBase(BaseModel):
    def dict(self, *args, **kwargs):
        return self.model_dump(*args, **kwargs)


def _model_from_typed_dict(typed_dict_cls, name: str = "BuildStateModel"):
    annotations = get_type_hints(typed_dict_cls)
    fields = {}
    for key, tp in annotations.items():
        default = _default_for_type(tp)
        fields[key] = (tp, default)
    return create_model(name, __base__=CompatBase, **fields)

BuildStateModel = _model_from_typed_dict(BuildState)

T = TypeVar('T')


def typed_dict_to_model(data: dict, model_cls: Type[T]) -> T:
    return model_cls(**data)


def model_to_typed_dict(model_instance) -> dict:
    return model_instance.model_dump()


def _safe_equal(a, b) -> bool:
    try:
        return a == b
    except Exception:
        return False


def typed_dict_changes(old_state: Mapping[str, Any], new_state: Mapping[str, Any], include_removed: bool = False) -> dict:
    """
    Compute a shallow diff between two TypedDict-like mappings.

    - Returns a dict that contains only the keys whose values changed, with their new values.
    - Keys added in new_state are included with their new values.
    - Keys removed from new_state are omitted by default; set include_removed=True to include them with value None.

    This uses Python equality (==), so it works for lists and nested structures where structural
    equality is appropriate. It does not attempt a recursive per-key deep diff; it only replaces
    entire values for changed keys.
    """
    changes: dict = {}
    keys = set(old_state.keys()) | set(new_state.keys())
    for k in keys:
        old_v = old_state.get(k, Ellipsis)
        new_v = new_state.get(k, Ellipsis)
        if new_v is Ellipsis:
            if include_removed:
                changes[k] = None
        elif old_v is Ellipsis:
            changes[k] = new_v
        elif not _safe_equal(old_v, new_v):
            changes[k] = new_v
    return changes





if __name__ == "__main__":
    example_dict: BuildState = {
        "Task_definition": "Create a Graph",
        "graph_rag_examples": ["alpha", "beta"],
        "graph_max_attempts": 3,
        "graph_attempt": 1,
        "graph_code_draft": ["print('hello')"],
        "graph_review_notes": ["looks good"],
        "graph_reviewer_agent_approved": False,
        "graph_exe_notes": [""],
        "graph_exe_agent_approved": True,
        "human_approved": False,
        "human_notes": "N/A",
    }
    example_dict =dict()
    
    model_instance = typed_dict_to_model(example_dict, BuildStateModel), 
    print("Pydantic model instance:", model_instance)

    roundtrip = model_to_typed_dict(model_instance)
    print("Back to dict:", roundtrip)

    for k, v in example_dict.items():
        assert roundtrip[k] == v, f"Mismatch on key '{k}': {roundtrip[k]} != {v}"
    print("Round-trip conversion succeeded.")

    empty_model = BuildStateModel()
    empty_dict = empty_model.model_dump()
    print("Empty model with defaults:", empty_dict)

    # Demo of typed_dict_changes
    old_state: BuildState = {
        "graph_rag_examples": ["alpha", "beta"],
        "graph_attempt": 1,
    }
    new_state: BuildState = {
        "graph_rag_examples": ["alpha", "beta", "gamma"],
        "graph_attempt": 1,
    }
    print("Changed fields:", typed_dict_changes(old_state, new_state))

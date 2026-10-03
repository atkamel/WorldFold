"""Teachers. Imported lazily so `PolicyTeacher` (pure torch) loads without a simulator: the scripted
teacher pulls in MuJoCo's expert, which the Isaac venv doesn't have (Phase I)."""

__all__ = ["Teacher", "ScriptedTeacher", "PolicyTeacher"]

_HOME = {"Teacher": "imitation.teachers.base", "ScriptedTeacher": "imitation.teachers.scripted",
         "PolicyTeacher": "imitation.teachers.policy"}


def __getattr__(name):
    if name in _HOME:
        import importlib
        return getattr(importlib.import_module(_HOME[name]), name)
    raise AttributeError(name)

# Lazy, so an Isaac Sim process (no mujoco installed) can import one teacher without the others.
__all__ = ["Teacher", "ScriptedTeacher", "PolicyTeacher", "IsaacScriptedTeacher"]
_MODULES = {"Teacher": "base", "ScriptedTeacher": "scripted", "PolicyTeacher": "policy",
            "IsaacScriptedTeacher": "isaac"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    import importlib
    return getattr(importlib.import_module(f"imitation.teachers.{_MODULES[name]}"), name)

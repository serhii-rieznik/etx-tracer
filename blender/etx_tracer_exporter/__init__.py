bl_info = {
    "name": "ETX Tracer Scene Exporter",
    "author": "ETX Tracer Community",
    "version": (2, 2, 0),
    "blender": (4, 4, 0),
    "location": "File > Export",
    "description": "Export complete ETX Tracer scene (JSON + OBJ + Materials)",
    "doc_url": "",
    "tracker_url": "",
    "category": "Import-Export",
    "support": "COMMUNITY",
}

import importlib
from . import operator
from . import logic


def register():
    importlib.reload(operator)
    importlib.reload(logic)
    operator.register()


def unregister():
    operator.unregister()


if __name__ == "__main__":
    register()

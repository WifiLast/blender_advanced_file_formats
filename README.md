# Navisworks & Revit Importer (Blender add-on)

Adds **File > Import > Navisworks (.nwd)** and **Revit (.rvt)** and drag-and-drop of both file types into the 3D viewport.
No Autodesk software is needed: the readers are pure Python and run inside Blender
(numpy is bundled with Blender; `olefile` is vendored).

```
nwd/      nwd_tool (container), nwd_fragments (shapes/transforms/materials), nwd_geometry (type-94 records),
          nwd_graph (hierarchy), reader.py (-> placed meshes)
rvt/      reader.py (Revit elements -> meshes, metres, Z up)
vendor/   rvt_file_format (Revit database reader), olefile
operators/ import_nwd.py, import_rvt.py, _threaded.py (worker thread + modal timer, ESC cancels)
```

## Install
Zip the `blender_open` folder (the folder itself must be the zip's root content, i.e. `blender_manifest.toml` at the top) and use
*Edit > Preferences > Add-ons > Install from Disk*. Blender 4.2+.

## Behaviour
- **NWD**: one object per fragment, identical geometry shares one mesh, colour/transparency from the NWD material. Scale option (NWD = metres, Z up).
  Triangle-strip and line records are decoded; other forms (quantised, UV, points) are skipped and reported.
  Transform type 17 (full matrix) is not decoded yet (treated as identity). Object names come from the hierarchy only when the path numbering lines up, else `Item_<path>`.
- **RVT**: reads the 3D geometry stored in the file (walls, floors, families, MEP, ...), one object per element, collection per category,
  Revit materials, parameters as custom properties. Optional category regex. Large projects need minutes and >1 GB RAM (93 MB Golden Nugget ~7-8 min).
  2D/annotation elements are skipped.

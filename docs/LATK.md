# Lightning Artist Toolkit (Latk) Format & Architecture Guide

This document provides a comprehensive specification and implementation guide for the **Lightning Artist Toolkit (Latk)** 3D brushstroke file format. It is optimized for AI agents interacting with Latk files across various programming languages.

## 1. Latk JSON Format Specification

The Latk format represents 3D spatial drawings and volumetric animations over time. The JSON structure is hierarchical, heavily inspired by Blender's Grease Pencil.

### JSON Structure
```json
{
    "creator": "latk.py",
    "version": 2.9,
    "grease_pencil": [
        {
            "layers": [
                {
                    "name": "GP_Layer",
                    "frames": [
                        {
                            "strokes": [
                                {
                                    "color": [ 0.20259166, 0.032980658, 0.9169371, 1.0 ],
                                    "fill_color": [ 0.0, 0.0, 0.0, 1.0 ],
                                    "brush_name": "optional",
                                    "brush_creator": "optional",
                                    "points": [
                                        {
                                            "co": [ 1.1935601, 0.98816276, -0.74828625 ], 
                                            "pressure": 0.50230646, 
                                            "strength": 0.50914043, 
                                            "vertex_color": [ 0.0, 0.0, 0.0, 1.0 ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}
```

### Spec Hierarchy Details
* **Root Object**: Contains metadata (`creator`, `version`) and the main `grease_pencil` array.
* **`grease_pencil`**: Array containing animation project data.
* **`layers`**: Array of layer objects (e.g., for separating colors or elements).
* **`frames`**: Array of frames within a layer, representing a snapshot in time.
* **`strokes`**: A continuous line drawn by the user. Includes RGBA arrays for `color` and `fill_color`.
* **`points`**: Array of vertices. `co` is a 3D coordinate `[x, y, z]`. Also stores `pressure`, `strength`, and `vertex_color`.

---

## 2. Universal Core Data Model

Across all language implementations, Latk libraries follow a standard Object-Oriented hierarchy:

1. **`Latk`**: The root container. Manages the animation timeline, file I/O (reads/writes `.json` and zipped `.latk` formats), and holds a list of `LatkLayer`s.
2. **`LatkLayer`**: Represents a single layer. Contains a sequence of `LatkFrame`s and tracks the `currentFrame`.
3. **`LatkFrame`**: Represents a single frame of animation in a timeline sequence. Contains a list of `LatkStroke` instances.
4. **`LatkStroke`**: The atomic visual element representing a continuous 3D line. Contains a list of `LatkPoint` instances and styling properties (color/size). Usually provides methods for stroke modification (smoothing, splitting, refining, cleaning).
5. **`LatkPoint`**: The fundamental vertex unit. Holds the 3D coordinate (`co`/`PVector`/`ofVec3f`), pressure, strength, and color.

---

## 3. Language Implementations

### Python (`latkpy`)
* **Use Case:** Headless processing, geometric optimization, scripting.
* **File I/O:** Supports standard `.json` and `.latk` zipped operations via `InMemoryZip` (`latk_zip.py`). Also parses Google Tilt Brush (`.tilt`) files (`latk_tilt.py`).
* **Geometry Processing:**
  * `latk_rdp.py`: Ramer-Douglas-Peucker algorithm for stroke simplification.
  * `latk_kmeans.py`: K-means clustering for spatial optimization and color quantization.

### C++ / openFrameworks (`ofxLatk`)
* **Use Case:** High-performance rendering, native desktop/embedded applications.
* **Dependencies:** Bundled with `JsonCpp` and a lightweight `zip` library (no external Poco dependency).
* **Execution Flow:** `Latk::run()` checks elapsed time against framerate (`checkInterval()`), increments `currentFrame` on `LatkLayer`s, and tells strokes to update.
* **Rendering:** Applications access the active `LatkStroke` points to draw paths or meshes.

### JavaScript (`latk.js`)
* **Use Case:** Web environments (Three.js, p5.js, 2D Canvas).
* **Dependencies:** Bundled with `JSZip` for unzipping `.latk`, `.sketch` (Tilt Brush), and Oculus Quill archives.
* **Processing Utilities:** Includes methods like `clean(epsilon)` (RDP simplification), `normalize()`, `refine()`, `smoothStroke()`, and `splitStroke()`.

### Java / Processing (`latkProcessing`)
* **Use Case:** Creative coding in the Processing IDE.
* **Data Flow:** Integrates directly into the Processing `draw()` loop. Calling `latk.run()` cascades down the hierarchy to render internally cached `PShape` objects at the stroke level.
* **Importers:** Contains `TiltLoader` and `QuillLoader` for converting binary formats into the native Latk structure.

### C# / Unity (`latkUnity`)
* **Use Case:** Real-time game engines, VR/AR, Unity projects.
* **Architecture:** Driven by a central `LightningArtist.cs` MonoBehaviour.
* **Rendering System:** Abstracted via `LatkStrokeRenderer`. The default is `LatkLineRenderer` (uses Unity's `LineRenderer`), allowing developers to implement custom shaders, ribbon meshes, or particles easily.
* **Modules:** Segregated into `Importers` (Tilt/Quill), `Drawing` (primitives generation), `Input` (keyboard, mouse, VR controllers), and `Playback` (syncing with Unity's Animator/Audio/Video).

---

## 4. Agent Guidelines for Latk Operations

When an AI agent is tasked with generating or manipulating Latk files, adhere to these rules:

1. **Hierarchy Integrity:** Always respect the 5-tier nested structure (`Latk` > `Layer` > `Frame` > `Stroke` > `Point`). Skipping a level will corrupt the file.
2. **File Formats:** By default, save data as flat `.json` for debugging or direct text manipulation. For production, compress the `.json` into a `.zip` and rename the extension to `.latk`. The libraries handle both automatically.
3. **Point Reduction:** 3D drawing data is dense. Use RDP algorithms (`clean()` or `latk_rdp.py`) when transferring strokes to lower file sizes and improve runtime performance.
4. **Coordinate Normalization:** When mixing sources (e.g., Tilt Brush with Latk), utilize the built-in `normalize()` functions to scale points into a unified 0-1 bounding box.
5. **Timeline Handling:** Animations are driven by `LatkLayer`s traversing `LatkFrame` arrays. If creating a static 3D drawing (non-animated), place all `LatkStroke`s inside a single `LatkFrame` (index 0).

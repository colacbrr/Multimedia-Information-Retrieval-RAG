# Local Video Dataset

Place local video clips here when running the video retrieval extension.

Recommended layout:

```text
data/videos/
  personal/
    clip_001.mp4
  public/
    sample_001.mp4
  annotations/
    captions.json
```

Supported video extensions include `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`.

The optional `annotations/captions.json` file can be either a list or a mapping. Example:

```json
[
  {
    "video_id": "public__sample_001",
    "file_name": "public/sample_001.mp4",
    "captions": ["A person walking through a city street."]
  }
]
```

Generated frames, embeddings, FAISS indexes, uploads, and metrics are written to `backend/outputs/video/` and are intentionally ignored by Git.

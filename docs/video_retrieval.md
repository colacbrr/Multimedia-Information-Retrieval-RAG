# Video Retrieval

This extension adds CLI and API support for text-to-video retrieval, frame-level retrieval, and video-to-text reverse retrieval.

## What It Adds

- OpenCV-based video ingestion and frame sampling
- CLIP embeddings for sampled frames
- video-level vectors created by pooling frame embeddings
- FAISS indexes for video, frame, and caption retrieval
- optional caption reranking
- timestamp and best-frame metadata for temporal grounding
- reverse retrieval from an indexed video ID, local video path, or uploaded clip
- optional Ollama summaries over video retrieval results

## Data Layout

The repo keeps only placeholders and metadata examples in Git. Put local clips under:

```text
data/videos/
  personal/
  public/
  annotations/
    captions.json
```

`captions.json` is optional. When present, it enables caption reranking and stronger reverse retrieval.

Example:

```json
[
  {
    "video_id": "public__sample_001",
    "file_name": "public/sample_001.mp4",
    "captions": ["A person walking through a city street."]
  }
]
```

## Build The Index

From the repo root:

```bash
python scripts/build_video_index.py --videos data/videos --mode both --sampling uniform --fps 1
```

Generated artifacts are written to `backend/outputs/video/`:

- extracted frames
- frame embeddings
- video embeddings
- caption embeddings
- FAISS indexes
- manifest and metrics logs

These files are ignored by Git.

## Text-To-Video Search

```bash
python scripts/search_video.py --query "people walking in a city" --top-k 5 --mode video
python scripts/search_video.py --query "a close up of food" --top-k 5 --mode frame
```

The API equivalent:

```bash
curl "http://localhost:8000/video/search?query=people%20walking%20in%20a%20city&top_k=5&mode=video"
```

## Video-To-Text Reverse Retrieval

From the API:

```bash
curl -X POST "http://localhost:8000/video/reverse-search" \
  -H "Content-Type: application/json" \
  -d '{ "video_path": "data/videos/public/sample_001.mp4", "top_k": 5 }'
```

For a terminal workflow:

```bash
python scripts/terminal_video_demo.py
```

## Evaluation

If captions are available, run:

```bash
python scripts/evaluate_video_retrieval.py --captions data/videos/annotations/captions.json --top-k 10
```

The script stores `evaluation_summary.json` in the configured video output directory.

## Limitations

- frame sampling can miss short actions
- caption reranking depends on annotation quality
- large video datasets can create many extracted frames
- reverse retrieval works best when caption embeddings are available
- local Ollama summaries are slower than retrieval

# Semantic Multimedia Retrieval System with Grounded Explanations

This project is a local-first multimedia retrieval system for image and video search. It turns natural-language queries into ranked visual results, supports video-to-text retrieval from local clips, and generates grounded explanations over the retrieved evidence.

It combines:

- CLIP embeddings for shared text-image and text-video semantic space
- FAISS indexing for fast nearest-neighbor retrieval
- caption-aware reranking for better result ordering
- a local Ollama explanation layer over retrieved items
- benchmark artifacts for latency and retrieval quality analysis
- CLI tools for text-to-video retrieval, video-to-text retrieval, and evaluation

## What Problem This Solves

Traditional keyword search is a poor fit for images and videos. Users ask for meaning, scenes, actions, and concepts, while media files usually expose only weak metadata.

This project addresses that semantic gap by:

1. encoding text queries, images, and sampled video frames into compatible embedding spaces
2. retrieving semantically similar images, videos, or video frames with vector search
3. optionally reranking results using caption similarity
4. generating an explanation from the retrieved evidence rather than from free-form hallucinated context

The result is a system that is more useful than raw vector search alone because it returns both ranked matches and a grounded explanation of what was found.

## What The System Does

Given a query such as `a dog playing in the park`, the system can search either images or videos:

1. encodes the query with CLIP
2. searches a FAISS index built from image embeddings, video-level embeddings, or frame-level embeddings
3. attaches captions, timestamps, frame previews, and metadata to the retrieved media
4. optionally reranks results using caption embeddings
5. sends the final retrieved context to a local Ollama model
6. returns a structured explanation with a summary, uncertainty hint, and referenced retrieved items

The video extension also supports reverse retrieval: a local clip is sampled, embedded, and matched against indexed caption/text evidence to produce video-to-text results and an optional summary.

## Why This Is Interesting

This is not only a UI demo. It is an end-to-end retrieval workflow with measurable tradeoffs:

- retrieval is fast enough for interactive use
- explanation is slower and becomes the dominant latency cost
- explanation quality depends directly on retrieval quality and caption coverage
- the system is modular enough to extend toward richer multimodal pipelines

## Architecture

```text
query
  -> CLIP text encoder
  -> FAISS vector retrieval over image, video, or frame embeddings
  -> optional caption reranking
  -> retrieved context assembly
  -> local Ollama explanation
  -> structured response + metrics logging
```

Main layers:

- Embedding layer: CLIP text, image, and sampled-frame encoders
- Retrieval layer: FAISS flat or HNSW indexes for images, videos, frames, and captions
- Reranking layer: caption similarity fusion
- Explanation layer: local retrieval-grounded generation
- Evaluation layer: stored benchmarks and runtime metrics

More detail:

- [Architecture](docs/architecture.md)
- [Startup](docs/startup.md)
- [Demo Guide](docs/demo-guide.md)
- [Results](docs/results.md)
- [Workflow](docs/workflow.md)
- [Video Retrieval](docs/video_retrieval.md)
- [Research Notes](docs/research-notes.md)

## Recommended Project Title

If you want a descriptive public title, use:

`Semantic Multimedia Retrieval System with Grounded Explanations`

## Current Scope

Implemented:

- image retrieval
- text-to-image semantic search
- video retrieval
- text-to-video semantic search
- frame-level video search with timestamp hints
- video-to-text reverse retrieval for local clips
- FAISS flat and HNSW support
- caption-aware reranking
- local explanation generation with Ollama
- prompt versioning
- explanation caching
- fallback handling for malformed model output
- benchmark and runtime metrics

Planned:

- audio retrieval
- richer hybrid search
- stronger reranking
- broader video evaluation

## Complexity and Tradeoffs

This project sits at the intersection of multimodal retrieval and generation, so its complexity is mostly systems complexity rather than UI complexity.

Main complexity points:

- embedding generation is compute-heavy and front-loaded
- vector retrieval is fast but quality-sensitive to encoder choice and index size
- video indexing requires frame extraction and can grow quickly with sampling rate
- caption reranking improves precision but adds extra inference work
- explanation depends on retrieved context quality
- local LLM generation is much slower than retrieval and needs careful timeout, caching, and fallback behavior

The explanation layer is retrieval-grounded, but it is not a full document-chunk RAG system yet. There is no separate text chunk store, no citation graph, and no long-context retrieval stage.

## Benchmark Snapshot

Stored benchmark artifacts currently show:

### 1k image run

- average query latency: 6.92 ms
- recall@1: 0.46
- recall@5: 0.755
- recall@10: 0.88

### 5k image run

- average query latency: 6.9 ms
- recall@1: 0.315
- recall@5: 0.48
- recall@10: 0.57

### Explanation summary

- average search time: 68.07 ms
- average LLM time: 13737.42 ms

These numbers matter because they make the tradeoff visible: retrieval remains interactive, while generation is the expensive stage.

## Repository Layout

```text
backend/     FastAPI retrieval and explanation service
frontend/    React + Vite interface
scripts/     CLI tools for video indexing, search, reverse retrieval, and evaluation
docs/        architecture, startup, results, workflow
data/videos/ local video dataset placeholder, ignored except docs/metadata stubs
assets/      screenshots and diagrams
benchmarks/  stored evaluation outputs
tests/       focused helper tests
```

## Installation

Full setup instructions live in [docs/startup.md](docs/startup.md). Short version:

### Requirements

- Python 3.10+
- Node.js 18+
- Ollama

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python search_server.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Ollama

Install Ollama, start the local service, then pull a model such as:

```bash
ollama pull llama3.1:8b
```

The explanation layer will run only if Ollama is available and the selected model exists locally.

### Dataset

The repository does not include media datasets. For image retrieval, the backend expects:

```text
backend/val2017/
backend/annotations/captions_val2017.json
```

You can download the required files directly from the official COCO dataset mirrors:

```bash
cd backend

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip
```

After extraction, the backend should contain:

```text
backend/
  val2017/
  annotations/
    captions_val2017.json
```

Only `val2017/` and `annotations/captions_val2017.json` are required for the current image retrieval workflow.

For video retrieval, place clips under:

```text
data/videos/
  personal/
  public/
  annotations/captions.json
```

Captions are optional for visual search, but they improve reranking and enable stronger video-to-text reverse retrieval.

Build and search the video index from the repo root:

```bash
python scripts/build_video_index.py --videos data/videos --mode both
python scripts/search_video.py --query "people walking in a city" --top-k 5 --mode video
python scripts/terminal_video_demo.py
```

## API Quickstart

Search:

```bash
curl "http://localhost:8000/search?query=a%20dog%20playing%20in%20the%20park&top_k=5"
```

Video search:

```bash
curl "http://localhost:8000/video/search?query=people%20walking%20in%20a%20city&top_k=5&mode=video"
curl "http://localhost:8000/search?query=people%20walking%20in%20a%20city&top_k=5&modality=video"
```

Video-to-text reverse retrieval from a local file path:

```bash
curl -X POST "http://localhost:8000/video/reverse-search" \
  -H "Content-Type: application/json" \
  -d '{ "video_path": "data/videos/public/sample_001.mp4", "top_k": 5 }'
```

Generate a retrieval-grounded explanation:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a dog playing in the park",
    "model": "llama3.1:8b",
    "results": [
      { "file_name": "000000000139.jpg", "caption": "A dog running through a grassy park.", "score": 0.51 }
    ]
  }'
```

Useful endpoints:

- `/status`
- `/search`
- `/rag`
- `/explain`
- `/video/status`
- `/video/search`
- `/video/reverse-search`
- `/video/rag`
- `/metrics`
- `/metrics/summary`
- `/benchmarks`
- `/ollama/models`

## Limitations

- media datasets and generated indexes are not bundled in the repo
- explanation quality is bounded by retrieval quality
- video search quality depends on frame sampling and available captions
- local generation is significantly slower than retrieval
- the current explanation layer is not yet a full citation-rich RAG stack

## Next Improvements

The most useful next steps for this repository are:

- embed screenshots directly into the root README
- add a small API smoke test for `/search` and `/explain`
- add a task runner such as a `Makefile` for setup, run, and test commands
- add Docker or `docker-compose` for reproducible local startup
- extend evaluation beyond retrieval latency and Recall@K
- expand the system toward audio retrieval

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

## Screenshots

Screenshots and diagrams are available under `assets/`.

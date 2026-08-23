# IM CLASSY

A real-time image classifier that runs entirely in the browser — no backend, no server, no images ever uploaded anywhere. Point your camera at an object and see live predictions, or upload a still image.

Built on [transformers.js](https://github.com/xenova/transformers.js) running `Xenova/vit-base-patch16-224` (a quantized ViT model) via WebAssembly/WebGPU.

## Running locally

Camera access requires a secure context, so opening `index.html` directly (`file://`) will not work for the camera tab (the upload tab still will). Serve the folder over `localhost` instead, e.g.:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` and allow camera access when prompted.

## Files

- `index.html` — markup
- `styles.css` — styling
- `app.js` — model loading, camera capture loop, and classification logic

# Website Deployment

This is a static HTML, CSS, and JavaScript website. It does not need a build command.

## GitHub Pages

1. Commit and push the project to the `main` branch.
2. In GitHub, open `Settings > Pages` for the repository.
3. Set the Pages source to `GitHub Actions`.
4. Wait for the `Deploy website to GitHub Pages` workflow to finish.
5. Share:

```text
https://logeshkhanna.github.io/lifetwin-ai/
```

The camera demo needs HTTPS or localhost. GitHub Pages provides HTTPS.

## Local preview

From the project folder, run:

```text
python -m http.server 4173
```

Then open `http://localhost:4173/`.

## Important

MediaPipe and OpenCV.js are loaded from their public CDNs when the interactive lab starts. A viewer therefore needs an internet connection for face/lip detection. The report does not provide trained CNN-LSTM weights or measured results, so the prediction and metrics placeholders remain honest until those assets are supplied.

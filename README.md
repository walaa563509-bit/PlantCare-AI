# PlantCare AI – App Starter

## Files needed
Place these files in the same GitHub repository as `app.py`:
- `PlantCare_AI_finetuned_best.keras`
- `PlantCare_AI_classes.json`

Do not upload the 2.33 GB dataset ZIP. The app only needs the trained model and class-name JSON for inference.

## Run locally / deploy
Install packages from `requirements.txt`, then run:
`streamlit run app.py`

For Streamlit Community Cloud, select `app.py` as the main file.

## Notes
- The app includes six sections and Arabic/English switching.
- Scan history is session-only in this starter version.
- Care advice is intentionally general and cautious.
- The model score is not a calibrated guarantee.

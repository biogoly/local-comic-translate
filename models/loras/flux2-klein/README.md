# FLUX.2 Klein LoRAs

Place compatible LoRA weights in the folder matching their base model:

- `4b/` for FLUX.2 Klein 4B
- `9b/` for FLUX.2 Klein 9B

Only `.safetensors` files are discovered. A weight may have an optional JSON
sidecar with the same filename stem, for example `lettering.safetensors` and
`lettering.json`:

```json
{
  "display_name": "Lettering Helper",
  "base_model": "klein-4b",
  "trigger_words": ["artistic lettering"],
  "default_scale": 0.8,
  "description": "Preserves hand-drawn title lettering."
}
```

`default_scale` must be between `0.0` and `2.0`. Model placement is
authoritative: a 4B LoRA is never shown while the 9B model is active, and vice
versa. Weight files are intentionally ignored by Git.

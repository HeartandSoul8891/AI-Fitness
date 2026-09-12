# AI-Fitness Textual Inversion

Files:
- `scripts/textual_inversion/textual_inversion_script.py`
- `scripts/textual_inversion/requirements.txt`
- `tabs/textual_inversion/embedding_tab.py`

The trainer uses Diffusers' SD1.x textual-inversion training pattern:
only the newly-added token embedding(s) are learned; VAE and UNet are frozen.
It supports CPU and PyTorch GPU runtimes, including ROCm because ROCm PyTorch
exposes the accelerator through `torch.cuda`.

Outputs include:
- `learned_embeds.bin`
- `learned_embeds.pt`
- optional `learned_embeds.safetensors`
- `<token>.pt`
- `token_identifier.txt`
- `type_of_concept.txt`
- `training_metadata.json`
- `config.json`
- checkpoint folders
- validation images when a validation prompt is configured

The UI uses the existing AI-Fitness settings paths for datasets and output.

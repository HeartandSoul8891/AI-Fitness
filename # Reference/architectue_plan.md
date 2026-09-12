Phase 1 — Rebuild the foundation
2. Rebuild Settings around modern model architecture

This is probably the most important architectural change.

Instead of settings assuming:

model = one .pt file

we want to think in terms of:

Model
├── Architecture
├── Diffusion model / UNet / DiT
├── Text encoder(s)
├── Tokenizer(s)
├── VAE
├── CLIP / conditioning components
├── Precision
└── Training compatibility

That gives us room for:

SD 1.5
SD 2.x
SDXL
Flux
SD3/3.5
future architectures

without redesigning the entire trainer every time.

And this matters enormously for Textual Inversion, because an embedding isn't simply a generic .pt file. It belongs to a particular tokenizer/text-encoder setup.

So the trainer should know something like:

Embedding
    ↓
Tokenizer
    ↓
Text Encoder
    ↓
Model Architecture

rather than treating an embedding as an isolated file.

Phase 2 — Clean project architecture
3. Split tabs

I'd absolutely do this:

tabs/
├── yolo/
│   ├── __init__.py
│   ├── trainer_tab.py
│   ├── annotator_tab.py
│   ├── auto_tagger_tab.py
│   └── dataset_tab.py
│
├── embedding/
│   ├── __init__.py
│   ├── trainer_tab.py
│   └── ...
│
├── lora/
│   ├── __init__.py
│   ├── trainer_tab.py
│   └── ...
│
├── hypernetwork/
│   └── ...
│
└── settings_tab.py

And eventually:

scripts/
├── yolo/
├── embedding/
├── lora/
├── hypernetwork/
└── common/

This is much cleaner than eventually ending up with:

scripts/
yolo_trainer_script.py
embedding_trainer_script.py
embedding_trainer_v2.py
embedding_trainer_final.py
embedding_trainer_final2.py
...

😂

Phase 3 — Dataset architecture
5. Rework datasets

I'd make this trainer-independent.

Something along the lines of:

datasets/
├── source/
│   ├── my_yolo_dataset/
│   └── my_embedding_dataset/
│
├── prepared/
│   ├── yolo/
│   └── embedding/
│
└── metadata/

But I'd go one step further.

The dataset should have metadata describing what it is:

dataset
├── images
├── captions
├── metadata
├── resolution
├── format
├── trainer compatibility
└── preprocessing state

Then the trainer consumes a dataset rather than manipulating random folders itself.

That makes the eventual UI much nicer too.

Phase 4 — Fix the YOLO workflow
6. Kill this:
datasets
   ↓
training
   ↓
runs

as a hard-coded workflow.

Instead:

Dataset
   ↓
Training Job
   ↓
Output

Where the training job owns its run.

For example:

Training/
└── yolo/
    └── run_2026-09-11_001/
        ├── config.json
        ├── logs/
        ├── checkpoints/
        ├── metrics/
        └── results/

And the dataset remains independent:

datasets/
└── my_dataset/

This is much more future-proof.

A dataset shouldn't suddenly become part of Training merely because you trained it.

Phase 5 — Launcher + installation
7. Rebuild launcher

Instead of:

Start_Yolo-Trainer.bat

eventually I'd like:

Start-Trainer.bat

which does something like:

Check Python
      ↓
Check environment
      ↓
Check GPU backend
      ↓
Load configuration
      ↓
Start Streamlit

And ideally the user doesn't need to know any of that.

8. Proper installer

This should come after the architecture stabilizes.

The installer should eventually handle:

Python/environment
dependencies
PyTorch
ROCm
model directories
datasets
configuration
shortcuts
launcher
first-run hardware detection

And critically:

don't install a gigantic pile of incompatible dependencies just because they're available.

The current AMD situation is a perfect example of why.

Phase 6 — GPU abstraction
9. ROCm support

I'd actually make this a backend abstraction, not an AMD hack.

You already have:

backend/
├── cpu_backend.py
├── cuda_backend.py
├── detect_backend.py
└── rocm_backend.py

That's the right idea.

I'd turn that into a proper common interface:

backend/
├── __init__.py
├── detect.py
├── base.py
├── cpu.py
├── cuda.py
└── rocm.py

Then trainers don't care whether they're running on:

CPU
CUDA
ROCm

They ask the backend:

device = backend.device
precision = backend.precision
memory = backend.memory

etc.

That will also make your RX 9070 XT situation much easier to manage.

Phase 7 — Finally: Textual Inversion
10. Build the actual Embedding / Textual Inversion trainer

And this is where I think your original idea becomes interesting.

I wouldn't try to recreate A1111.

I'd make:

A modern Textual Inversion trainer that happens to be able to produce A1111-compatible embeddings.

That's a very different goal.

The UI could eventually look something like:

╔══════════════════════════════════════════╗
║       TEXTUAL INVERSION TRAINER          ║
╠══════════════════════════════════════════╣
║                                          ║
║ BASE MODEL                               ║
║ [ SD 1.5 ▼ ]                             ║
║ [ model/checkpoint ................ ]    ║
║                                          ║
║ EMBEDDING                                ║
║ Placeholder token [ <myconcept>       ]  ║
║ Initializer token [ person            ]  ║
║ Vectors           [ 8                 ]  ║
║                                          ║
║ DATASET                                  ║
║ [ My Concept Dataset ▼ ]                 ║
║ Images: 42                               ║
║ Repeats: [ 10 ]                          ║
║                                          ║
║ TRAINING                                 ║
║ Batch size       [ 2 ]                   ║
║ Learning rate    [ 0.005 ]               ║
║ Scheduler        [ cosine ▼ ]            ║
║ Steps            [ 5000 ]                ║
║                                          ║
║ VALIDATION                               ║
║ Prompt:                                  ║
║ [ photo of <myconcept> ............... ] ║
║                                          ║
║ [ ▶ START TRAINING ]  [ ■ STOP ]        ║
║                                          ║
║ Step 1240 / 5000                         ║
║ Loss: 0.1842                             ║
║                                          ║
║ Validation images                        ║
║ [ image ] [ image ] [ image ] [ image ]  ║
╚══════════════════════════════════════════╝

But underneath it we'd have a real training pipeline.

And there's one particularly important thing I want to investigate

Before we write the Textual Inversion trainer, I'd like us to establish exactly what level of Textual Inversion we're targeting.

Because there are actually several interesting directions:

A1111-style
learn token embedding

Very simple.

Diffusers-style
tokenizer
     ↓
text encoder
     ↓
learnable embedding vectors
     ↓
diffusion training

More configurable.

Our eventual version

Potentially:

Dataset
   ↓
Caption / prompt processing
   ↓
Tokenizer
   ↓
Learnable vectors
   ↓
Text encoder
   ↓
Diffusion model
   ↓
Loss
   ↓
Optimizer
   ↓
Embedding
   ↓
Validation

with proper checkpointing and experiment configuration.

That's the version I'd like to build.

And importantly, we can still export the resulting embedding in the format expected by the ecosystems that support it.

So I would NOT start coding Textual Inversion yet

I'd first rebuild the skeleton:

AI-Fitness
│
├── backend/
│
├── datasets/
│
├── models/
│
├── training/
│
├── tabs/
│   ├── yolo/
│   ├── embedding/
│   └── lora/
│
├── scripts/
│   ├── yolo/
│   ├── embedding/
│   └── lora/
│
├── config/
│
├── outputs/
│
├── logs/
│
└── main.py

Then migrate the existing YOLO trainer without changing its functionality initially.

Once YOLO works on the new foundation, we have a stable platform on which to build the embedding trainer.

That avoids the classic trap of simultaneously changing:

architecture + dataset system + settings + GPU backend + YOLO + embedding training + installer

…and then having absolutely no idea which change broke what. 😅

I think this is the point where your “little trainer” can actually become a proper multi-trainer application.
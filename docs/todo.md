## TODOs

### General
- [ ] Add benchmarking script for evaluation
- [ ] Maybe try NVFP4 during Pretraining
- [ ] Try [LiquidAI ColBERT](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) instead of Gemma Embeddings

### Data
- [ ] Generate multiple reasoning paths per question using Gemma4
- [x] *Deprecated: possible consolidation of dataset files into chunks (eg. 3 files loaded)*
    > should allow for better grouping by similarity

### Architectural changes
- [x] Add Normalization and Dropout layers to the model architecture
- [x] Skip connections between experts
- [x] Add attention experts
    - [x] Related: Switch to Gemma4 for Per Layer Embeddings (also providing more contextual information)
    - [x] Add embedding layer input to all experts
    - [ ] Add consistent reasoning trace during post training (via MoE vector input)
- [ ] Expert top k routing
- [ ] Multi Token Prediction
- [ ] Move MoE to latent space with a skip connection

### Training
- [x] Pretraining script using Ultrafineweb v1.4
- [x] SFT script using KIMI-K2.5-550000x and reasoning dataset
    > SFT should feature finetuning all parameters.
    > 
    > Use `Chat` template for formatting the data.
- [x] Update Pre- and Posttraining scripts to accomadate changes in dataset architecture
- [x] Logging to files


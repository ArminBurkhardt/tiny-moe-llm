## TODOs

### General
- [ ] Add benchmarking script for evaluation

### Data
- [ ] Generate multiple reasoning paths per question using Gemma4
- [ ] possible consolidation of dataset files into chunks (eg. 3 files loaded)
    > should allow for better grouping by similarity

### Architectural changes
- [x] Add Normalization and Dropout layers to the model architecture
- [x] Skip connections between experts
- [ ] Add attention experts
    - [ ] Related: Switch to Gemma4 for Per Layer Embeddings (also providing more contextual information)
- [ ] Expert top k routing
- [ ] Multi Token Prediction
- [ ] Move MoE to latent space with a skip connection

### Training
- [x] Pretraining script using Ultrafineweb v1.4
- [x] SFT script using KIMI-K2.5-550000x and reasoning dataset
    > SFT should feature finetuning all parameters.
    > 
    > Use `Chat` template for formatting the data.
- [ ] Update Pre- and Posttraining scripts to accomadate changes in dataset architecture


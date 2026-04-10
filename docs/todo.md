## TODOs

- [x] Pretraining script using Ultrafineweb v1.4
- [x] SFT script using KIMI-K2.5-550000x and reasoning dataset
    > SFT should feature finetuning all parameters.
    > 
    > Use `Chat` template for formatting the data.
- [ ] Generate multiple reasoning paths per question using Gemma4 
- [x] Add Normalization and Dropout layers to the model architecture
- [x] Skip connections between experts
- [ ] Add attention experts
    - [ ] Related: Switch to Gemma4 for Per Layer Embeddings (also providing more contextual information)
- [ ] Expert top k routing
- [ ] possible consolidation of dataset files into chunks (eg. 3 files loaded)
    > should allow for better grouping by similarity
- [ ] Update Pre- and Posttraining scripts to accomadate changes in dataset architecture
- [ ] Add benchmarking script for evaluation
- [ ] Multi Token Prediction
 

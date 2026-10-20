## Study Plan for PathVQ    

Below is a suggested 7-day study plan to understand the PathVQ paper. Each day has a specific focus, with readings, exercises, and self-tests to reinforce your understanding.

### **Day 1: Math and neural net basics**

- Vectors, dot product, cosine similarity, matrix multiplication, softmax, linear layer, MLP, loss and backprop (idea only).
- Watch 3Blue1Brown's neural network series.
- *Self-test:* why does cosine similarity measure "how alike" two vectors? What does softmax output?

### **Day 2: Attention and the Transformer**

- Read Jay Alammar's "The Illustrated Transformer".
- Then work through the formula by hand with tiny 3×3 matrices: `softmax(QKᵀ/√d)·V`.
- Learn multi-head attention, residual connections, LayerNorm and the feed-forward block.
- *Self-test:* what are Q, K, V? Why divide by √d? Why is attention quadratic in sequence length?

### **Day 3: Vision Transformer and the [CLS] token**

- Read the ViT paper, "An Image is Worth 16x16 Words".
- Understand patches → tokens → Transformer → [CLS].
- This is the core of PathVQ's problem: a 224×224 tile gives 196 patch tokens plus one [CLS]. Everyone keeps only [CLS], and PathVQ argues that throws away information.
- *Self-test:* why is a tile 14×14 = 196 patches? What does [CLS] summarize?

### **Day 4: Self-supervised learning and foundation models**

- Learn masked modeling (BERT, MAE), DINO/iBOT at the idea level, and what UNI is (a tile encoder trained this way).
- Then read BEiT v2's idea: a *tokenizer* gives masked tokens discrete targets. PathVQ copies this.
- *Self-test:* what is a foundation model? What is "frozen" vs "fine-tuned"?

### **Day 5: Vector quantization (the heart of the paper)**

- Learn the autoencoder first: encoder → bottleneck → decoder.
- Then VQ-VAE (van den Oord 2017). Know these by heart:
  - codebook = a table of C learnable vectors
  - nearest-neighbor lookup: `z = argmin ‖e − v_j‖`
  - straight-through estimator (argmin has no gradient, so copy it)
  - reconstruction loss plus codebook loss
- Then residual and multi-scale quantization (VAR paper idea). This is their MSVQ.
- Compute the compression by hand: 1024 floats → one index into 8192 codes with a 16-d embedding, giving the paper's 64× (1024 → 16).
- *Self-test:* why quantize instead of just using a smaller MLP? What is "codebook collapse"?

### **Day 6: WSI and MIL**

- A slide gives thousands of tiles but only one label. That is Multiple Instance Learning.
- Read the ABMIL paper (Ilse 2018) and learn its attention pooling formula.
- *Self-test:* how does ABMIL turn N tile features into one slide vector? Why is attention used instead of averaging?

### **Day 7: Read PathVQ**

- Pass 1 (15 min): abstract, Fig. 1, Fig. 3, conclusion.
- Pass 2: Method 2.1–2.5, one equation at a time.
- Pass 3: experiments (Tables 1–2, Fig. 5), and note the limitations (needs extra pretraining, some information loss).
- Make a one-page answer sheet: **problem → why [CLS] fails → VQ compression → MSVQ → slide-level SSL → results**.

## **Formulas you should be able to write from memory**

1. Attention: `softmax(QKᵀ/√d)V`
2. Cosine similarity
3. VQ nearest neighbor and straight-through
4. ABMIL attention pooling: `a_i = softmax(wᵀ tanh(V h_i))`, `z = Σ a_i h_i`

For one-line answers to your professor's likely questions:

- **What did they do?** They compressed all patch tokens with VQ instead of keeping only [CLS].
- **Why?** [CLS] loses spatial detail, and keeping every token is ~200× more expensive.
- **How good?** Better F1 and AUC than [CLS]-only baselines, even beating UNI-2 on the paper's tasks.

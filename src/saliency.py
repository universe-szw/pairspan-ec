import torch
import numpy as np

def get_risk_heatmap(prompt, pipeline, tokenizer, max_length=200):
    """
    Computes a risk heatmap using gradient saliency mapping.
    Tokens whose embeddings contribute most to the 'harmful' classification
    (LABEL_0) get higher scores via L2 norm of the gradient.
    """
    model = pipeline.model
    device = model.device

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get embeddings and enable gradient tracking
    model.zero_grad()
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)

    # Forward pass with gradients enabled
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    
    # Backpropagate from the harmful logit (LABEL_0)
    harmful_logit = outputs.logits[0, 0]
    harmful_logit.backward()

    # L2 norm of gradient per token = saliency score
    risk_scores = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()

    # Get token strings for reference
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Strip [CLS] and [SEP]
    if len(tokens) > 2:
        risk_scores = risk_scores[1:-1]
        tokens = tokens[1:-1]
    elif len(tokens) == 2:
        risk_scores = risk_scores[1:1]
        tokens = []

    # Normalize to [0, 1]
    if np.max(risk_scores) - np.min(risk_scores) > 1e-9:
        risk_scores = (risk_scores - np.min(risk_scores)) / (np.max(risk_scores) - np.min(risk_scores))
    else:
        risk_scores = np.zeros_like(risk_scores)

    return risk_scores, tokens

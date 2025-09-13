# AI Model Setup

This directory should contain the Mistral 7B Instruct model for local inference.

## Required Model

**File:** `mistral-7b-instruct-v0.1.Q4_K_M.gguf` (4.2GB)

## Download Instructions

1. **Visit Hugging Face**: [Mistral 7B Instruct GGUF](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
2. **Download the GGUF file**: Look for `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
3. **Place in this directory**: `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`

## Alternative Sources

- **Direct download**: Use `wget` or browser download
- **Hugging Face Hub**: `huggingface-hub download mistralai/Mistral-7B-Instruct-v0.1`
- **Other GGUF variants**: Q4_K_M recommended for balance of speed/quality

## Verification

After download, verify the file:
- **Size**: ~4.2GB
- **Format**: `.gguf`
- **Location**: `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`

The application will automatically detect and load the model on startup.

## Note

This model is not included in the repository due to GitHub's file size limits. 
The download is required for AI generation functionality.
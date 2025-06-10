# LLM-finetuning

The core idea in this repo is not to just do RAG
But, implement improvements in every dimensions of it 
From indexing to generation all step by step...


![Overall RAG FLOW](/assessts/ragflow.drawio.png)

Exploration of LLM finetuning task on newer domain

|Scale of models exploration
| - SLMs
| - LLMs


Key Learning:

1) Multi-agent is just for formatting things and passing it to context better. So for example, if raw dollar values are passed as context for generation model, nothing happens, or it doesn't understand it well. But, say if this raw is passed through another generation model to format the context in plain english which in turn is passed as context, wallah you have multi-agent flow.

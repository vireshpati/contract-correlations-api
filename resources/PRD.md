Task: Create a FastAPI application that predicts how prediction market contracts correlate with each other.

Data:  contract_correlations table (see notebooks/contract_correlations.ipynb)

```
Loaded 750 rows from contract_correlations table
id                                      object
contract_a_id                           object
contract_a_title                        object
contract_a_venue                        object
contract_a_data                         object
contract_b_id                           object
contract_b_title                        object
contract_b_venue                        object
contract_b_data                         object
probability_correlation                float64
underlying_event_correlation           float64
correlation_type                        object
correlation_strength                    object
correlation_reasoning                   object
price_correlation                      float64
volume_correlation                     float64
temporal_alignment                      object
expiry_correlation                      object
event_category_match                      bool
common_factors                          object
causal_relationship                     object
analysis_confidence                    float64
anthropic_model                         object
analysis_prompt_version                 object
processing_time_ms                       int64
is_active                                 bool
needs_refresh                             bool
last_updated                    datetime64[ns]
created_at                      datetime64[ns]
dtype: object
```

Model: LLama 3.1 8B QLoRA fine-tuned on the contract_correlations table + RAG. From Hugging Face.

Expected Output Format:

{
    "underlying_correlation": 0.5,
    "correlation_type": "positive",
    "confidence": 0.95,
    "reasoning": "Both contracts are related to the same underlying event..."
}

Code Structure: All functional patterns. CONCISE, clean, modular. all files shoudl be created as scaffold.

Environment: you are writing code on a mac. all code will be ran on an H200 interactive GPU machine. Then deployed later on thundercompute.

Test: pytest





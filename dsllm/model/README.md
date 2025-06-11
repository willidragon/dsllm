# Model Compatibility Notes

## Transformers Version Compatibility

The model implementation includes a `cache_position` parameter that is only required for transformers versions >= 4.51.0. This parameter is used in the `forward` method of the `SensorLLMStage1LlamaModel` class.

### For transformers >= 4.51.0
- Keep the `cache_position` parameter in the `forward` method signature
- Keep the `cache_position` parameter in the `super().forward()` call

### For transformers < 4.51.0
- Remove the `cache_position` parameter from the `forward` method signature
- Remove the `cache_position` parameter from the `super().forward()` call

The relevant code sections are marked with comments in `stage1_sensorllm.py`:

```python
def forward(
    # ... other parameters ...
    # cache_position is only needed for transformers >= 4.51.0
    # Remove this parameter if using older versions
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # ... method implementation ...
    return super(SensorLLMStage1LlamaModel, self).forward(
        # ... other parameters ...
        # cache_position is only needed for transformers >= 4.51.0
        # Remove this parameter if using older versions
        cache_position=cache_position,
    )
```

## Version Check
To check your transformers version:
```bash
pip show transformers
``` 
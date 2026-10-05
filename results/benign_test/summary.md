# Benign-only vs refusal-only teachers (report §4.8)

Students at 20,000 samples, mean of 2 seeds. Rates are fractions; margin in logits.

## Teachers

```
metric     harmbench_harmful  harmbench_refusal  prefill_asr_k5  prefill_asr_k20  refusal_margin  xstest_overrefusal
teacher                                                                                                             
T_benign               0.250              0.725           0.850            0.925           8.976               0.048
T_none                 0.035              0.910           0.355            0.620          22.371               0.040
T_refusal              0.005              0.995           0.690            0.845          16.460               0.284
T_shallow              0.060              0.935           0.780            0.855          15.916               0.196
```

## Their students

```
metric     harmbench_harmful  harmbench_refusal  prefill_asr_k5  prefill_asr_k20  refusal_margin  xstest_overrefusal
teacher                                                                                                             
T_benign               0.143              0.840           0.782            0.870          10.336               0.054
T_none                 0.048              0.910           0.355            0.612          21.662               0.044
T_refusal              0.062              0.915           0.690            0.818          13.626               0.076
T_shallow              0.138              0.845           0.752            0.867           9.656               0.072
```

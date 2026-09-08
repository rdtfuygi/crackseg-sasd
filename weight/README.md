# Trained weights of SASD-CrackSeg

One best checkpoint per dataset (highest F1 over the five seeded runs reported in the paper). Each file is a PyTorch `state_dict` of `CrackSegSASD` (483,648 parameters).

| File | Dataset | Source run | F1 (val) |
|---|---|---|---|
| `deepcrack_best.pth` | DeepCrack | run 3 | 0.8765 |
| `steelcrack_best.pth` | SteelCrack | run 1 | 0.8763 |
| `ycd_best.pth` | YCD | run 0 | 0.8481 |
| `crack500_best.pth` | Crack500 | run 0 | 0.7979 |

Load:
```python
from crackseg_sasd import CrackSegSASD
model = CrackSegSASD(input_channel=3, output_channel=1)
model.load_state_dict(torch.load('deepcrack_best.pth', map_location='cpu'))
```

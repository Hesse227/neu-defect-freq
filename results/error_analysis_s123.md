# Error analysis (baseline vs +DCT, test split, conf=0.25, IoU 0.5)

Recall per class (box-level matching):

| class | GT boxes | baseline recall | +DCT recall |
|---|---|---|---|
| crazing | 64 | 0.406 | 0.156 |
| inclusion | 111 | 0.856 | 0.820 |
| patches | 91 | 0.857 | 0.868 |
| pitted_surface | 44 | 0.750 | 0.705 |
| rolled-in_scale | 72 | 0.639 | 0.625 |
| scratches | 52 | 0.942 | 0.942 |

GT boxes missed by baseline but caught by +DCT: **16**

- rolled-in_scale: 7
- inclusion: 4
- crazing: 3
- patches: 1
- scratches: 1

# Error analysis (baseline vs +DCT, test split, conf=0.25, IoU 0.5)

Recall per class (box-level matching):

| class | GT boxes | baseline recall | +DCT recall |
|---|---|---|---|
| crazing | 64 | 0.406 | 0.172 |
| inclusion | 111 | 0.856 | 0.757 |
| patches | 91 | 0.868 | 0.846 |
| pitted_surface | 44 | 0.727 | 0.705 |
| rolled-in_scale | 72 | 0.639 | 0.708 |
| scratches | 52 | 0.923 | 0.904 |

GT boxes missed by baseline but caught by +DCT: **20**

- rolled-in_scale: 15
- crazing: 2
- inclusion: 1
- pitted_surface: 1
- scratches: 1

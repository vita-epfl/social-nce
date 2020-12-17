## Social NCE: Contrastive Learning of Socially-aware Motion Representations

<p align="center">
  <img src="docs/illustration.png" width="300">
</p>

<!-- This repository contains the code and data to replicate the experiments in our paper:
```bibtex
@article{liu2020snce,
    title={Social NCE: Contrastive Learning of Socially-aware Motion Representations},
    author={Yuejiang Liu and Qi Yan and Alexandre Alahi},
    year={2020}
}
```
 -->

### Preparation

Setup environments follwoing the [SETUP.md](docs/SETUP.md)

### Training & Evaluation

* Behavioral Cloning (Vanilla)
  ```
  python imitate.py --contrast_weight=0.0
  python test.py --policy='sail' --circle --model_file=data/output/imitate-baseline-data-0.5/policy_net.pth
  ```
* Random Negative Sampling (Conventional)
  ```
  python imitate.py --contrast_weight=0.5 --contrast_sampling='local' --gpu
  python test.py --policy='sail' --circle --model_file=data/output/imitate-local-data-0.5-weight-0.5-horizon-4-temperature-0.20-nboundary-0-range-2.00/policy_net.pth
  ```
* Safety-driven Sampling (Ours)
  ```
  python imitate.py --contrast_weight=0.5 --contrast_sampling='event' --gpu
  python test.py --policy='sail' --circle --model_file=data/output/imitate-event-data-0.5-weight-0.5-horizon-4-temperature-0.20-nboundary-0/policy_net.pth
  ```

### Example Results

Behavioral cloning results obtained at the 100th epoch with different methods (NVIDIA RTX 2080 Ti GPU):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Vanilla</th>
<th valign="bottom">Random</th>
<th valign="bottom">Ours</th>
<!-- TABLE BODY -->
<tr><td align="left">Success</td>
<td align="center">0.88</td>
<td align="center">0.80</td>
<td align="center">0.98</td>
<tr><td align="left">Reward</td>
<td align="center">0.274</td>
<td align="center">0.240</td>
<td align="center">0.329</td>
</tr>
</tbody></table>

### Acknowledgments

Our code is developed based on [CrowdNav](https://github.com/vita-epfl/CrowdNav). 

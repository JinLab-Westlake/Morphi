<img
  src="./docs/images/Picture1.png"
  align="right"
  width="300"
  alt="Multi-dimensional organoid phenotype analysis platform"
/>

### Multi-dimensional organoid phenotype analysis platform

AI-powered organoid analysis platform integrating
z-stack projection, fluorescence quantification,
instance segmentation, and viability assessment.

[Documentation](链接) |
[Datasets](链接) |
[Paper](链接)

<br clear="right"/>

---

## Overview

简要介绍平台解决什么问题，以及主要应用场景。



## Repository Branches

This repository is organized into multiple branches corresponding to the complete Morphi platform and its individual analysis modules.

### 1. `Morphi`

The `Morphi` branch contains the complete integrated software platform. It combines image preprocessing, quantitative analysis, deep-learning-based segmentation, viability assessment, feature extraction, and result visualization within a unified workflow.

This branch is recommended for users who want to perform the full organoid image analysis pipeline.

### 2. Functional Module Branches

The following branches provide standalone implementations of the major components used in Morphi.

#### `Z-stack-projection`

Performs projection of multi-plane microscopy image stacks into representative two-dimensional images for downstream analysis.

#### `Fluorescence-quantification`

Quantifies fluorescence signals from microscopy images, including image-level or organoid-level intensity measurements and fluorescence-positive regions.

#### `DoANet`

Provides the DoANet-based viability assessment module. It identifies viable and non-viable regions and calculates viability-related measurements for individual organoid instances.

#### `Organoid-instance-segmentation`

Detects and segments individual organoids from microscopy images, enabling instance-level analysis rather than image-level aggregate measurements.

#### `Multi-dimensional-feature-extraction`

Extracts quantitative features describing organoid morphology, fluorescence characteristics, viability, and other image-derived phenotypes.


## Installation

安装方法。

## Quick Start

最简运行示例。

## Documentation

详细文档入口。

## Datasets

训练集、测试集或示例数据说明。

## Citation

论文发表后的引用信息。

## License

代码许可证。

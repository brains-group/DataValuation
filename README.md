# DataValuation
Faizaan Ali, Oshani Seneviratne

# Project Overview

This project aims to develop a generalizable, domain-agnostic definition of data quality, with a focus on label correctness as a core component. While many existing quality metrics are domain-specific or tied to assumptions about data structure, our goal is to build methods that extend across diverse tabular datasets.

# Current Work

We are building a dataset-adaptive ensemble meta-model for mislabel detection.
The model combines multiple complementary detectors—global classifier confidence, local neighborhood disagreement, and density-based outlier scores—and learns how to weight them based on dataset-level statistics such as dimensionality, feature distribution, class imbalance, separability, etc.

To train this system, we generate controlled noisy versions of benchmark datasets that include ground-truth mislabels, allowing us to evaluate detector performance and teach the meta-model how to adapt to different noise profiles.

# Long-Term Goal

By establishing a reliable, general-purpose measure of label quality, this work aims to support broader efforts in data valuation, dataset benchmarking, and quality-aware machine learning pipelines.

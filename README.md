# Contextualized Sparse Representations for Real-Time Open-Domain Question Answering

This repository contains the implementation of an ICLR submission [Contextualized Sparse Representation with Rectified N-Gram Attention for Open-Domain Question Answering](https://openreview.net/forum?id=ryxgegBKwr).
The code is based on [DenSPI repository](https://github.com/uwnlp/denspi) where detailed information on the requirements and the installation is available.

See `modeling.py` for the implementation of `CoSPR` (defined as the class [`SparseAttention`](https://github.com/jhyuklee/CoSPR/blob/e659a16775633aacad94317ecc72fa6278890952/modeling.py#L228)) and its [kernelized training](https://github.com/jhyuklee/CoSPR/blob/e659a16775633aacad94317ecc72fa6278890952/modeling.py#L908).

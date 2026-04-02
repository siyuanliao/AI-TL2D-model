# AI-TL2D-model
This document presents the Python code for two baseline models, detailing their network architectures and training procedures.

Background: In 2026, we submitted a manuscript to JASA describing a lightweight two-dimensional propagation loss prediction model named LCMF-Net. To highlight the model’s performance and lightweight characteristics, we proposed two baseline models for comparison based on the classic U‑Net and conditional GAN architectures, referred to as U‑Net‑2D and GAN‑2D, respectively.

Detailed descriptions of the two baseline models are as follows:

(1) Baseline Model A: U‑Net‑2D
This model is based on the classic U‑Net architecture and has approximately 4.43 million parameters. It adopts a symmetric encoder–decoder design in which multi‑scale features are fused through skip connections. Environmental parameters are encoded into a conditional vector by an MLP and then concatenated with the historical‑mean field as the network input. The encoder comprises four downsampling stages to extract deep features, while the decoder consists of four upsampling stages to gradually restore spatial resolution. Skip connections between corresponding encoder and decoder layers facilitate the fusion of low‑level and high‑level features, representing a typical application of encoder–decoder methods in acoustic field reconstruction.

(2) Baseline Model B: Conditional GAN (GAN‑2D)
This model is built on the generative adversarial network framework, with the generator having about 3.74 million parameters. The generator is composed of 16 residual blocks that incorporate FiLM conditional modulation and dilated convolutions to expand the receptive field. The discriminator takes both the predicted acoustic field and the environmental parameters as inputs to perform conditional discrimination. The training objective combines adversarial loss with L1 reconstruction loss to enhance both the distributional realism and structural consistency of the predicted fields. This model exemplifies the typical paradigm of applying generative adversarial training to physical field prediction tasks.

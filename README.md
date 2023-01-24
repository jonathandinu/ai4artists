[![simple neural net art diagram](images/simple-neural-net-art-diagram.png)](https://teia.art/objkt/538875)

> Resources at the intersection of AI _AND_ Art. Mainly tools and tutorials but also with some inspiring people and places thrown in too!

For a broader resource covering more general creative coding tools (that you might want to use with what is listed here), check out [terkelg/awesome-creative-coding](https://github.com/terkelg/awesome-creative-coding) or [thatcreativecode.page](https://thatcreativecode.page/). For resources on AI and deep learning in general, check out [ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning) and [https://github.com/dair-ai](https://github.com/dair-ai).

## Contents

* [Learning](#learning)
  * [Courses](#courses)
  * [Videos](#videos)
  * [Books](#books)
  * [Tutorials and Blogs](#tutorials-and-blogs)
* [Papers/Methods](#papers-methods)
  * [Diffusion models (and text-to-image)](#diffusion-models-and-text-to-image) 
  * [Neural Radiance fields (and NeRF like things)](#neural-radiance-fields-and-nerf-like-things)
  * [3D and point clouds](#3d-and-point-clouds)
  * [Unconditional Image Synthesis](#unconditional-image-synthesis)
  * [Conditional Image Synthesis (and inverse problems)](#conditional-image-synthesis-and-inverse-problems)
  * [GAN inversion (and editing)](#gan-inversion-and-editing)
  * [Latent Space Interpretation](#latent-space-interpretation)
  * [Image Matting](#image-matting)
* [Tools](#tools)
  * [Creative ML](#creative-ml)
  * [Deep Learning](#deep-learning-frameworks)
  * [Runtimes/Deployment](#runtimesdeployment)
  * [text-to-image](#text-to-image)
  * [Creative Coding](#creative-coding)
  * [Stable Diffusion](#stable-diffusion-sd)
* [Datasets](#datasets)
* [Products/Apps](#productsapps)
* [Artists](#artists)
* [Institutions/Places](#institutionsplaces)
* [Related Lists](#related-lists-and-collections)

> __bold__ entries signify my favorite resource(s) for that section/subsection (if I _HAD_ to choose a single resource). Additionally each subsection is usually ordered by specificity of content (most general listed first).

## Learning

### Courses

#### General Deep Learning

* [Practical Deep Learning for Coders (fast.ai)](https://course19.fast.ai/index.html)
* [Deep Learning (NYU)](https://atcold.github.io/pytorch-Deep-Learning/)
* [Introduction to Deep Learning (CMU)](https://deeplearning.cs.cmu.edu/F22/resources.html)
* ⭐️ __[Deep Learning for Computer Vision (UMich)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)__
* [Deep Learning for Computer Vision (Stanford CS231n)](http://cs231n.stanford.edu/index.html)
* [Natural Language Processing with Deep Learning (Stanford CS224n)](https://web.stanford.edu/class/cs224n/)

#### Deep Generative Modeling

* [Deep Generative Models (Stanford)](https://deepgenerativemodels.github.io/)
* [Deep Unsupervised Learning (UC Berkeley)](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
* [Differentiable Inference and Generative Models (Toronto)](http://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html)
* ⭐️ __[Learning-Based Image Synthesis (CMU)](https://learning-image-synthesis.github.io/sp22/)__
* [Learning Discrete Latent Structure (Toronto)](https://duvenaud.github.io/learn-discrete/)
* [From Deep Learning Foundations to Stable Diffusion (fast.ai)](https://www.fast.ai/posts/part2-2022.html)

#### Creative Coding and New Media

* ⭐️ __[Deep Learning for Art, Aesthetics, and Creativity (MIT)](https://ali-design.github.io/deepcreativity/)__
* [Machine Learning for the Web (ITP/NYU)](https://github.com/yining1023/machine-learning-for-the-web)
* [Art and Machine Learning (CMU)](https://sites.google.com/site/artml2018/lectures)
* [New Media Installation: Art that Learns (CMU)](https://artthatlearns.wordpress.com/syllabus/)
* Introduction to Computational Media (ITP/NYU)
  * [Media course](https://github.com/ITPNYU/ICM-2022-Media)
  * [Code course](https://github.com/ITPNYU/ICM-2022-Code)

### Videos

* ⭐️ __[The AI that creates any picture you want, explained (Vox)](https://youtu.be/SVcsDDABEkM)__
* [I Created a Neural Network and Tried Teaching it to Recognize Doodles (Sebastian Lague)](https://youtu.be/hfMk-kjRv4c)
* [Neural Network Series (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [Beginner's Guide to Machine Learning in JavaScript (Coding Train)](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6YPSwT06y_AEYTqIwbeam3y)
* [Two Minute Papers](https://www.youtube.com/c/K%C3%A1rolyZsolnai)

### Books

* ⭐️ __[Dive into Deep Learning (Zhang, Lipton, Li, and Smola)](https://d2l.ai/index.html)__
* [Deep Learning (Goodfellow, Bengio, and Courville)](https://www.deeplearningbook.org/)
* [Computer Vision: Algorithms and Applications (Szeliski)](https://szeliski.org/Book/)
* [Procedural Content Generation in Games (Shaker, Togelius, and Nelson)](http://pcgbook.com/)
* [Generative Design (Benedikt Groß)](http://www.generative-gestaltung.de/2/)

### Tutorials and Blogs

#### Deep Learning

* ⭐️ __[VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance (Crowson and Biderman)](https://arxiv.org/pdf/2204.08583.pdf)__
* [Tutorial on Deep Generative Models (IJCAI-ECAI 2018)](https://ermongroup.github.io/generative-models/)
* [Tutorial on GANs (CVPR 2018)](https://sites.google.com/view/cvpr2018tutorialongans/)
* [Lil'Log (Lilian Weng)](https://lilianweng.github.io/)
* [Distill [on hiatus]](https://distill.pub/)

#### Generative Art

* ⭐️ __[Making Generative Art with Simple Mathematics](http://www.hailpixel.com/articles/generative-art-simple-mathematics)__
* [Book of Shaders: Generative Designs](https://thebookofshaders.com/10/)
* [Mike Bostock: Visualizing Algorithms](https://bost.ocks.org/mike/algorithms/) (with [Eyeo talk](https://vimeo.com/112319901))
* [Generative Examples in Processing](https://github.com/digitalcoleman/generativeExamples)
* [Generative Music](https://teropa.info/loop/#/title)

### Papers/Methods

#### Diffusion models (and text-to-image)

* [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073): Paper predating Stable Diffusion describing a method for image synthesis and editing with diffusion based models.
* [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741.pdf)
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html): Original paper that introduced Stable Diffusion and started it all.
* [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io): Edit Stable Diffusion outputs by editing the original prompt.
* [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://textual-inversion.github.io): Similar to prompt-to-prompt but instead takes an input image and a text description.  Kinda like Style Transfer... but with Stable diffusion.
* [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://dreambooth.github.io): Similar to Textual Inversion but instead focused on manipulating subject based images (i.e. _this thing/person/etc._ but *underwater*).
* [Novel View Synthesis with Diffusion Models](https://arxiv.org/abs/2210.04628)
* [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352)
* [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://makeavideo.studio)
* [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276)
* [MDM: Human Motion Diffusion Model](https://guytevet.github.io/mdm-page/)
* [Soft Diffusion: Score Matching for General Corruptions](https://arxiv.org/abs/2209.05442)
* [Multi-Concept Customization of Text-to-Image Diffusion](https://www.cs.cmu.edu/~custom-diffusion/): Like DreamBooth but capable of synthesizing multiple concepts.
* [eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://deepimagination.cc/eDiff-I/)
* [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://github.com/NVlabs/edm)
* [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://nvlabs.github.io/denoising-diffusion-gan/)
* [Imagen Video: High Definition Video Generation with Diffusion Models](https://imagen.research.google/video/)

#### Neural Radiance fields (and NeRF like things)

* [Structure-from-Motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf): prior work on sparse modeling (still needed/useful for NeRF)
* [Pixelwise View Selection for Unstructured
Multi-View Stereo](https://demuc.de/papers/schoenberger2016mvs.pdf): prior work on dense modeling (NeRF kinda replaces this)
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103)
* [Deferred Neural Rendering: Image Synthesis using Neural Textures](https://arxiv.org/abs/1904.12356)
* [Neural Volumes: Learning Dynamic Renderable Volumes from Images](https://arxiv.org/abs/1906.07751)
* ⭐️ __[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)__: The paper that started it all...
* [Neural Radiance Fields for Unconstrained Photo Collections](https://nerf-w.github.io): NeRF in the wild (alternative to MVS)
* [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io): Photorealistic NeRF from casual in-the-wild photos and videos (like from a cellphone)
* [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://arxiv.org/abs/2103.13415): NeRF... but BETTER FASTER HARDER STRONGER
* [Depth-supervised NeRF: Fewer Views and Faster Training for Free](https://www.cs.cmu.edu/~dsnerf/): Train NeRF models faster with fewer images by leveraging depth information
* [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/): caching for NeRF training to make it rlllly FAST
* [Understanding Pure CLIP Guidance for Voxel Grid NeRF Models](https://hanhung.github.io/PureCLIPNeRF/): text-to-3D using CLIP
* [NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields](https://arxiv.org/abs/2210.13641): NeRF for robots (and cars)
* [nerf2nerf: Pairwise Registration of Neural Radiance Fields](https://nerf2nerf.github.io): pretrained NeRF
* [The One Where They Reconstructed 3D Humans and Environments in TV Shows](http://ethanweber.me/sitcoms3D/)
* [ClimateNeRF: Physically-based Neural Rendering for Extreme Climate Synthesis](https://climatenerf.github.io)
* [Realistic one-shot mesh-based head avatars](https://samsunglabs.github.io/rome/)
* [Neural Point Catacaustics for Novel-View Synthesis of Reflections](https://arxiv.org/pdf/2301.01087.pdf)
* [3D Moments from Near-Duplicate Photos](https://3d-moments.github.io)
* [NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors](https://arxiv.org/pdf/2212.03267.pdf)

### 3D and point clouds

* [DreamFusion: Text-to-3D using 2D Diffusion (Google)](https://dreamfusion3d.github.io)
* [ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding (Salesforce)](https://tycho-xue.github.io/ULIP/)
* [Extracting Triangular 3D Models, Materials, and Lighting From Images (NVIDIA)](https://nvlabs.github.io/nvdiffrec/)
* [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images (NVIDIA)](https://nv-tlabs.github.io/GET3D/)
* [3D Neural Field Generation using Triplane Diffusion](https://jryanshue.com/nfd/)
* [🎠 MagicPony: Learning Articulated 3D Animals in the Wild](https://3dmagicpony.github.io)
* [ObjectStitch: Generative Object Compositing (Adobe)](https://arxiv.org/pdf/2212.00932.pdf)
* [LADIS: Language Disentanglement for 3D Shape Editing (Snap)](https://arxiv.org/pdf/2212.05011.pdf)
* [Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion (Microsoft)](https://3d-avatar-diffusion.microsoft.com)
* [SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation (Snap)](https://yccyenchicheng.github.io/SDFusion/)
* [DiffRF: Rendering-guided 3D Radiance Field Diffusion (Meta)](https://sirwyver.github.io/DiffRF/)
* [Novel View Synthesis with Diffusion Models (Google)](https://3d-diffusion.github.io)
* ⭐️ __[Magic3D: High-Resolution Text-to-3D Content Creation (NVIDIA)](https://deepimagination.cc/Magic3D/)__

### Unconditional Image Synthesis

* [Sampling Generative Networks](https://arxiv.org/pdf/1609.04468.pdf)
* [Neural Discrete Representation Learning (VQVAE)](https://arxiv.org/abs/1711.00937)
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
* [A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)](https://arxiv.org/abs/1812.04948)
* ⭐️ __[Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2)](https://arxiv.org/abs/1912.04958)__
* [Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA)](https://github.com/NVlabs/stylegan2-ada-pytorch)
* [Alias-Free Generative Adversarial Networks (StyleGAN3)](https://github.com/NVlabs/stylegan3)
* [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf)
* [Taming Transformers for High-Resolution Image Synthesis (VQGAN)](https://compvis.github.io/taming-transformers/)
* [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)
* [StyleNAT: Giving Each Head a New Perspective](https://arxiv.org/pdf/2211.05770v1.pdf)
* [StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://sites.google.com/view/stylegan-xl/?pli=1)

### Conditional Image Synthesis (and inverse problems)

* [Image-to-Image Translation with Conditional Adversarial Nets (pix2pix)](https://phillipi.github.io/pix2pix/)
* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)](https://junyanz.github.io/CycleGAN/)
* [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (pix2pixHD)](https://tcwang0509.github.io/pix2pixHD/)
* [Semantic Editing of Scenes by Adding, Manipulating or Erasing Objects (SESAME)](https://arxiv.org/pdf/2004.04977v2.pdf)
* [Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)](https://github.com/NVlabs/SPADE)
* [You Only Need Adversarial Supervision for Semantic Image Synthesis (OASIS)](https://arxiv.org/pdf/2012.04781v3.pdf)
* [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://eladrich.github.io/pixel2style2pixel/)
* [Multimodal Conditional Image Synthesis with Product-of-Experts GANs](https://deepimagination.cc/PoE-GAN/)
* [Palette: Image-to-Image Diffusion Models](https://diffusion-palette.github.io)
* [Sketch-Guided Text-to-Image Diffusion Models](https://sketch-guided-diffusion.github.io)
* [HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation](https://github.com/lhoyer/HRDA)
* [PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation](https://arxiv.org/pdf/2211.07609v1.pdf)
* [MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation](https://github.com/lhoyer/MIC)
* [Pretraining is All You Need for Image-to-Image Translation (PITI)](https://tengfei-wang.github.io/PITI/index.html)

### GAN inversion (and editing)

* [Generative Visual Manipulation on the Natural Image Manifold (iGAN)](https://www.cs.cmu.edu/~junyanz/projects/gvm/)
* [In-Domain GAN Inversion for Real Image Editing](https://genforce.github.io/idinvert/)
* [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/pdf/1904.03189.pdf)
* [Designing an Encoder for StyleGAN Image Manipulation](https://github.com/omertov/encoder4editing)
* [Pivotal Tuning for Latent-based Editing of Real Images](https://github.com/danielroich/PTI)
* ⭐️ __[HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing](https://yuval-alaluf.github.io/hyperstyle/)__
* [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://github.com/orpatashnik/StyleCLIP)
* [High-Fidelity GAN Inversion for Image Attribute Editing](https://tengfei-wang.github.io/HFGI/)
* [Swapping Autoencoder for Deep Image Manipulation](https://taesung.me/SwappingAutoencoder/)
* [Sketch Your Own GAN](https://peterwang512.github.io/GANSketching/)
* [Rewriting Geometric Rules of a GAN](https://peterwang512.github.io/GANWarping/)
* [Anycost GANs for Interactive Image Synthesis and Editing](https://hanlab.mit.edu/projects/anycost-gan/)
* [Third Time’s the Charm? Image and Video Editing with StyleGAN3](https://yuval-alaluf.github.io/stylegan3-editing/)

### Latent Space Interpretation

* ⭐️ __[Discovering Interpretable GAN Controls (GANspace)](https://github.com/harskish/ganspace)__
* [Interpreting the Latent Space of GANs for Semantic Face Editing](https://genforce.github.io/interfacegan/)
* [GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](https://gandissect.csail.mit.edu)
* [Unsupervised Extraction of StyleGAN Edit Directions (CLIP2StyleGAN)](https://github.com/RameenAbdal/CLIP2StyleGAN)
* [Seeing What a GAN Cannot Generate](http://ganseeing.csail.mit.edu)

### Image Matting

* [Deep Image Matting](https://arxiv.org/pdf/1703.03872v3.pdf)
* [Background Matting: The World is Your Green Screen](https://arxiv.org/pdf/2004.00626v2.pdf)
* [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
* [Semantic Image Matting](https://arxiv.org/pdf/2104.08201v1.pdf)
* [Privacy-Preserving Portrait Matting](https://arxiv.org/pdf/2104.14222v2.pdf)
* [Deep Automatic Natural Image Matting](https://arxiv.org/pdf/2107.07235v1.pdf)
* [MatteFormer](https://github.com/webtoon/matteformer)
* [MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition](https://github.com/ZHKKKe/MODNet)
* ⭐️ __[Robust Human Matting via Semantic Guidance](https://arxiv.org/pdf/2210.05210v1.pdf)__

## Tools

### Generative Modeling

* [NVIDIA Imaginaire](https://github.com/NVlabs/imaginaire): 2D Image synthesis library
* [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/): The platform for creating and operating metaverse applications
* [mmgeneration](https://github.com/open-mmlab/mmgeneration)
* [Modelverse](https://modelverse.cs.cmu.edu): Content-Based Search for Deep Generative Models
* [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)

### Creative ML

* [Tensorflow.js](https://js.tensorflow.org/)
* [ml5.js](https://ml5js.org/)
* [MediaPipe](https://google.github.io/mediapipe/)
* ⭐️ __[Magenta](https://magenta.tensorflow.org/)__
* [Wekinator](http://www.wekinator.org/)
* [ofxAddons](http://ofxaddons.com/categories/14-machine-learning)

### Deep Learning Frameworks

* ⭐️ __[PyTorch](https://pytorch.org/)__
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [🤗 Transformers](https://huggingface.co/docs/transformers/index)
* [🤗 Diffusers](https://github.com/huggingface/diffusers)
* [JAX](https://github.com/google/jax)
* [dlib](http://dlib.net/)
* [Darknet](https://pjreddie.com/darknet/)

### Runtimes/Deployment

* [FFCV: an Optimized Data Pipeline for Accelerating ML Training](https://ffcv.io)
* [ONNX Runtime](https://onnxruntime.ai)
* [DeepSpeed (training, inference, compression)](https://github.com/microsoft/DeepSpeed)
* [TensorRT](https://developer.nvidia.com/tensorrt)
* [Tensorflow Lite](https://www.tensorflow.org/lite)
* [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
* [TorchServe](https://pytorch.org/serve/)
* [AITemplate](https://github.com/facebookincubator/AITemplate)

### Text-to-Image

* ⭐️ __[Stable Diffusion](https://github.com/CompVis/stable-diffusion)__
* [Imagen](https://github.com/lucidrains/imagen-pytorch)
* [DALLE 2](https://github.com/lucidrains/DALLE2-pytorch)
* [VQGAN+CLIP](https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks)
* [Parti](https://github.com/google-research/parti)
* [Muse: Text-To-Image Generation via Masked Generative Transformers](https://muse-model.github.io): More efficient than diffusion or autoregressive text-to-image models used masked image modeling w/ transformers

### Stable Diffusion (SD)

* [Dream Studio](https://beta.dreamstudio.ai/): Official [Stability AI](https://stability.ai) cloud hosted service.
* ⭐️ __[Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)__: A user friendly UI for SD with additional [features](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features) to make common workflows easy.
* [AI render (Blender)](https://airender.gumroad.com/l/ai-render): Render scenes in Blender using a text prompt.
* [Dream Textures (Blender)](https://github.com/carson-katri/dream-textures): Plugin to render textures, reference images, and background with SD.
* [lexica.art](https://lexica.art/) - SD Prompt Search. 
* [koi (Krita)](https://github.com/nousr/koi): SD plugin for [Krita](https://krita.org/en/) for img2img generation.
* [Alpaca (Photoshop)](https://www.getalpaca.io/): Photoshop plugin (beta).
* [Christian Cantrell's Plugin (Photoshop)](https://exchange.adobe.com/apps/cc/114117da/stable-diffusion): Another Photoshop plugin.
* [Stable Diffusion Studio](https://github.com/amotile/stable-diffusion-studio): Animation focused frontend for SD.
* [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII): Low-latency and high-throughput inference for a variety (20,000+) models/tasks, including SD.

### Neural Radiance Fields

* [COLMAP](https://colmap.github.io/index.html)
* ⭐️ __[nerfstudio](https://docs.nerf.studio/en/latest/index.html)__
* [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp)
* [NerfAcc](https://www.nerfacc.com/en/latest/index.html)

### Creative Coding

#### Frameworks

* ⭐️ __[Processing (Java)](https://processing.org/) and [p5.js (Javascript)](https://p5js.org/)__
* [openFrameworks (C++)](http://openframeworks.cc/) 
* [Cinder (C++)](https://libcinder.org/)
* [nannou (Rust)](https://nannou.cc/)

#### Visual Programming Languages

* [vvvv](https://vvvv.org/)
* ⭐️ __[TouchDesigner](https://derivative.ca/)__
* [Max/MSP/Jitter](https://cycling74.com/products/max/)
* [Pure Data](https://puredata.info/)

<!-- 
#### Audio

* [Sonic Pi](https://sonic-pi.net/)
* ⭐️ __[SuperCollider](https://supercollider.github.io/)__
* [Overtone](https://overtone.github.io/)
* [Tone.js](https://tonejs.github.io/)
* [ChucK](http://chuck.cs.princeton.edu/)

#### WebGL

> For a much much more extensive list see [sjfricke/awesome-webgl](https://github.com/sjfricke/awesome-webgl) 

* ⭐️ __[three.js](https://threejs.org/)__
* [regl](http://regl.party/)
* [stack.gl](http://stack.gl/) 

#### Text and NLP

* ⭐️ __[spacy](https://spacy.io/)__
* [natural](https://github.com/NaturalNode/natural)
* [Tracery](http://tracery.io/)


## Research

### Foundational Papers

- [Learning Deep Generative Models (Salakhutdinov 2015)](https://www.cs.cmu.edu/~rsalakhu/papers/annrev.pdf)
- [Generative Adversarial Networks (Goodfellow 2015)]
- [Autoregressive]
- [Variational Autoencioders (Kingma 2014)]
- [Flow]()
- [Diffusion (Sacha)]
 -->

## Datasets

* [LAION Datasets](https://github.com/LAION-AI/laion-datasets): Various very large scale image-text pairs datasets (notably used to train the open source [Stable Diffusion](https://stability.ai) models)
* [Unsplash Images](https://unsplash.com/data)
* [Open Images](https://storage.googleapis.com/openimages/web/index.html): Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives:
* [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets): 17,127 validated hours of transcribed speech covering 104 languages. Additionally many of the recorded hours in the dataset also include demographic metadata like age, sex, and accent that can help improve the accuracy of speech recognition engines.

### Faces/People

* [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [LFWA+](https://liuziwei7.github.io/projects/FaceAttributes.html)
* [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
* [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
* [UTKFace](https://susanqq.github.io/UTKFace/)
* [SSHQ](https://stylegan-human.github.io/data.html): full body 1024 x 512px

### Video

* [Brutus Light Field](https://github.com/nathan-fairchild/Brutus-Light-Fields)

## Products/Apps

* [Artbreeder](https://www.artbreeder.com/)
* [Midjourney](https://www.midjourney.com/)
* [DALLE 2 (OpenAI)](https://openai.com/dall-e-2/)
* [Runway](https://runwayml.com/) - AI powered video editor.
* [Facet AI](https://facet.ai/) - AI powered image editor.
* [Adobe Sensei](https://www.adobe.com/sensei/creative-cloud-artificial-intelligence.html) - AI powered features for the Creative Cloud suite.
* [NVIDIA AI Demos](https://www.nvidia.com/en-us/research/ai-demos/)
* [ClipDrop](https://clipdrop.co/) and [cleanup.pictures](https://cleanup.pictures/)

## Artists

A non-exhaustive list of people doing interesting things at the intersection of art, ML, and design.

- [Neural Bricolage (helena sarin)](https://twitter.com/NeuralBricolage)
- [Sofia Crespo](https://sofiacrespo.com/)
- [Anna Ridler](http://annaridler.com/)
- [Mimi Onuoha](http://mimionuoha.com/)
- [Claire Silver](https://www.clairesilver.com)
- [Sasha Stiles](https://www.sashastiles.com)
- [Ivona Tau](https://ivonatau.com)
- [Tega Brain](http://tegabrain.com/)
- [Lauren McCarthy](http://lauren-mccarthy.com/)
- [Allison Parrish](https://www.decontextualize.com/)
- [Caroline Sinders](https://carolinesinders.com/)
- [Memo Akten](http://www.memo.tv/works/)
- [Mario Klingemann](http://quasimondo.com/)
- [Trevor Paglen](https://paglen.studio/)
- [Tom White](https://drib.net/)
- [Robbie Barrat](https://robbiebarrat.github.io/)
- [Kyle McDonald](http://kylemcdonald.net/)
- [Golan Levin](http://www.flong.com/)

## Institutions/Places

- [STUDIO for Creative Inquiry](http://studioforcreativeinquiry.org/)
- [ITP @ NYU](https://tisch.nyu.edu/itp)
- [Gray Area Foundation for the Arts](https://grayarea.org/)
- [Stability AI (Eleuther, LAION, et al.)](https://stability.ai/)
- [Goldsmiths @ University of London](https://www.gold.ac.uk/computing/)
- [UCLA Design Media Arts](http://dma.ucla.edu/)
- [Berkeley Center for New Media](http://bcnm.berkeley.edu/)
- [Google Artists and Machine Intelligence](https://ami.withgoogle.com/)
- [Google Creative Lab](https://www.creativelab5.com/)
- [The Lab at the Google Cultural Institute](https://www.google.com/culturalinstitute/thelab/)
- Sony CSL ([Tokyo](https://www.sonycsl.co.jp/) and [Paris](https://csl.sony.fr/))

## Related lists and collections

* [Machine Learning for Art](https://ml4a.net/)
* [Tools and Resources for AI Art (pharmapsychotic)](https://pharmapsychotic.com/tools.html) - Big list of Google Colab notebooks for generative text-to-image techniques as well as general tools and resources.
* [Awesome Generative Deep Art](https://github.com/filipecalegario/awesome-generative-deep-art/blob/main/README.md) - A curated list of Generative Deep Art / Generative AI projects, tools, artworks, and models

## Contributing

Contributions are welcome! Read the [contribution guidelines](contributing.md) first.

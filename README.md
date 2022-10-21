# Awesome AI Art [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![simple neural net art diagram](images/simple-neural-net-art-diagram.png)](https://teia.art/objkt/538875)

> Resources at the intersection of AI _AND_ Art. Mainly tools and tutorials but also with some inspiring people and places thrown in too!

For a broader resource covering more general creative coding tools (that you might want to use with what is listed here), check out [terkelg/awesome-creative-coding](https://github.com/terkelg/awesome-creative-coding) or [thatcreativecode.page](https://thatcreativecode.page/). For resources on AI and deep learning in general, check out [ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning) and [https://github.com/dair-ai](https://github.com/dair-ai).

## Contents

* [Learning](#learning)
  * [Courses](#courses)
  * [Videos](#videos)
  * [Books](#books)
  * [Tutorials and Blogs](#tutorials-and-blogs)
  * [Papers](#papers)
* [Tools](#tools)
  * [Creative ML](#creative-ml)
  * [Deep Learning](#deep-learning-frameworks)
  * [text-to-image](#text-to-image)
  * [Creative Coding](#creative-coding)
  * [Stable Diffusion](#stable-diffusion-sd)
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

### Papers

* [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073): Paper predating Stable Diffusion describing a method for image synthesis and editing with diffusion based models.
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html): Original paper that introduced Stable Diffusion and started it all.
* [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io): Edit Stable Diffusion outputs by editing the original prompt.
* [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://textual-inversion.github.io): Similar to prompt-to-prompt but instead takes an input image and a text description.  Kinda like Style Transfer... but with Stable diffusion.
* [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://dreambooth.github.io): Similar to Textual Inversion but instead focused on manipulating subject based images (i.e. _this thing/person/etc._ but *underwater*).
* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io)
* [Novel View Synthesis with Diffusion Models](https://arxiv.org/abs/2210.04628)
* [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352)
* [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://makeavideo.studio)

## Tools

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

### Text-to-Image

* ⭐️ __[Stable Diffusion](https://github.com/CompVis/stable-diffusion)__
* [Imagen](https://github.com/lucidrains/imagen-pytorch)
* [DALLE 2](https://github.com/lucidrains/DALLE2-pytorch)
* [VQGAN+CLIP](https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks)
* [Parti](https://github.com/google-research/parti)

### Stable Diffusion (SD)

* [Dream Studio](https://beta.dreamstudio.ai/): Official [Stability AI](https://stability.ai) cloud hosted service.
* ⭐️ __[Stable Diffusion UI](https://github.com/cmdr2/stable-diffusion-ui)__: One click install for SD on a local machine with a user friendly UI.
* [AI render (Blender)](https://airender.gumroad.com/l/ai-render): Render scenes in Blender using a text prompt.
* [Dream Textures (Blender)](https://github.com/carson-katri/dream-textures): Plugin to render textures, reference images, and background with SD.
* [lexica.art](https://lexica.art/) - SD Prompt Search. 
* [koi (Krita)](https://github.com/nousr/koi): SD plugin for [Krita](https://krita.org/en/) for img2img generation.
* [Alpaca (Photoshop)](https://www.getalpaca.io/): Photoshop plugin (beta).
* [Christian Cantrell's Plugin (Photoshop)](https://exchange.adobe.com/apps/cc/114117da/stable-diffusion): Another Photoshop plugin.
* [Stable Diffusion Studio](https://github.com/amotile/stable-diffusion-studio): Animation focused frontend for SD.


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

## Products/Apps

* [Artbreeder](https://www.artbreeder.com/)
* [DALLE 2 (OpenAI)](https://openai.com/dall-e-2/)
* [Runway](https://runwayml.com/) - AI powered video editor.
* [Facet AI](https://facet.ai/) - AI powered image editor.
* [Adobe Sensei](https://www.adobe.com/sensei/creative-cloud-artificial-intelligence.html) - AI powered features for the Creative Cloud suite.
* [NVIDIA AI Demos](https://www.nvidia.com/en-us/research/ai-demos/)
* [ClipDrop](https://clipdrop.co/) and [cleanup.pictures](https://cleanup.pictures/)

## Artists

A non-exhaustive list of people doing interesting things at the intersection of art, ML, and design.

- [Neural Bricolage (helena sarin)](https://twitter.com/NeuralBricolage)
- [Memo Akten](http://www.memo.tv/works/)
- [Sofia Crespo](https://sofiacrespo.com/)
- [Mario Klingemann](http://quasimondo.com/)
- [Anna Ridler](http://annaridler.com/)
- [Trevor Paglen](https://paglen.studio/)
- [Tom White](https://drib.net/)
- [Mimi Onuoha](http://mimionuoha.com/)
- [Refik Anadol](http://refikanadol.com/)
- [Robbie Barrat](https://robbiebarrat.github.io/)
- [Gene Kogan](http://genekogan.com/)
- [Tega Brain](http://tegabrain.com/)
- [Lauren McCarthy](http://lauren-mccarthy.com/)
- [Kyle McDonald](http://kylemcdonald.net/)
- [Allison Parrish](https://www.decontextualize.com/)
- [Caroline Sinders](https://carolinesinders.com/)
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

## Contributing

Contributions are welcome! Read the [contribution guidelines](contributing.md) first.

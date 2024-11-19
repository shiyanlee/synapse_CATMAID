🚩**What was the problem?**
Ground truth annotations of neuronal structures are not only valuable for connectome reconstruction but also critical for training machine learning models in neuron segmentations. However, this is greatly distinctive in the context of synapse detection. Segmenting a neuron is challenging due to its complex trajectory across the tissue layers, while identifying a synapse is complicated by the variable cues at the presynaptic and postsynaptic site required for accurate classification.
In high-resolution EM images of the Drosophila brain, synapses are identified by darkly stained active zone proteins known as the T bar, alongside the postsynaptic density (PSD) complex which often appears as a ‘teeth-like’ structure. One of the challenges in developing synapse detection model is to distinguish synapses from other electron-dense organelles and artefacts. Additionally, the extremely thin slices captured with FIB-SEM implicates that the presynaptic and postsynaptic features may be captured at slightly different planes and distances, with variability across synapses. This inconsistency introduces unique perspectives on the synaptic structure, highlighting the inherent 3D complexity that complicates automated detection. 

⚔️**The approach and how it can be overcome.** 
CATMAID is an interface designed for navigating and annotating high resolution image stacks, providing a tool to create skeletonization of neurons and its respective synapses. Manually annotated synapses on CATMAID creates three types of annotations which is helpful for model training: 
-	Connectors, placed on the T-bar
-	Presynaptic neuron, indicating the neuron from which the synapse originates
-	Postsynaptic partners, annotated on the slice of the postsynaptic neuron where the PSD appears most prominent.
Using pymaid (a Python-CATMAID interface), this data curation pipeline enables the retrieval of specific pre- and postsynaptic partners without the need for involved neurons to be skeletonized. It also provides precise coordinates of these partners within the EM volume alongside the connectors. This allows the model to explicitly capture features of both presynaptic and postsynaptic neurons, even when each annotation of is situated at different location in 3D space. Then, connectors and their corresponding partners are encapsulated in a bounding box to proceed for data training. 




### 🚩**What was the problem and why is it important**
Ground truth annotations of neuronal structures are not only valuable for connectome reconstruction but also critical for training machine learning models in neuron segmentations. However, this is greatly distinctive in the context of synapse detection. Segmenting a neuron is challenging due to its complex trajectory across the tissue layers, while identifying a synapse is complicated by the variable cues at the presynaptic and postsynaptic site required for accurate classification.
In high-resolution EM images of the Drosophila brain, synapses are identified by darkly stained active zone proteins known as the T bar, alongside the postsynaptic density (PSD) complex which often appears as a ‘teeth-like’ structure. One of the challenges in developing synapse detection model is to distinguish synapses from other electron-dense organelles and artefacts. Additionally, the extremely thin slices captured with FIB-SEM implicates that the presynaptic and postsynaptic features may be captured at slightly different planes and distances, with variability across synapses. This inconsistency introduces unique perspectives on the synaptic structure, highlighting the inherent 3D complexity that complicates automated detection. 

### ⚔️**Our approach and how it can be overcome.** 
CATMAID is an interface designed for navigating and annotating high resolution image stacks, providing a tool to create skeletonization of neurons and its respective synapses. Manually annotated synapses on CATMAID creates three types of annotations which is helpful for model training: 
-	Connectors, placed on the T-bar
-	Presynaptic neuron, indicating the neuron from which the synapse originates
-	Postsynaptic partners, annotated on the slice of the postsynaptic neuron where the PSD appears most prominent.

Using pymaid (a Python-CATMAID interface), this data curation pipeline enables the retrieval of specific pre- and postsynaptic partners without the need for involved neurons to be skeletonized. It also provides precise coordinates of these partners within the EM volume alongside the connectors. This allows the model to explicitly capture features of both presynaptic and postsynaptic neurons, even when each annotation of is situated at different location in 3D space. Then, connectors and their corresponding partners are encapsulated in a bounding box to proceed for data training. 

### **📦 The Bounding Boxes**  

In this pipeline, synapses are only considered for training if **all connectors**, **pre-synaptic neurons**, and **post-synaptic neurons** are fully contained within a bounding box. 
#### **Two Case Examples of Bounding Box Creation:**  
1. **Synapses Traced in a Designated Cube:**  
This scenario involves a straightforward approach. The smallest and largest **x**, **y**, and **z** coordinates across all connectors, pre-synaptic neurons, and post-synaptic neurons are identified. These bounds ensure that no annotations fall outside the bounding box, encompassing all relevant data within a single, well-defined cube.  

2. **Synapses Traced in a Single Kenyon Cell:**  
In this case, the connectors are distributed irregularly and non-uniformly. To address this complexity:  
- **DBSCAN Clustering** was applied to group points based on proximity, limiting the spatial spread and forming multiple smaller clusters.  
- For each cluster, compact bounding boxes were created using the cluster points.  

#### **Challenges and Solutions:**  
- To account for the spread of elements in the synapse, the bounding boxes are stretched, which expands the dataset and introduces empty spaces to ensure that all relevant components—connectors, pre-, and post-synaptic neurons—are fully encapsulated.
- The strong irregularity and small dataset made the application of shrinkage factors for bounding boxes suboptimal. Shrinkage factors apply uniform reductions to all axes, which was too rigid for this dataset.  
- To overcome, bounding dimensions were manually defined to allow greater flexibility, focusing on regions with the strongest density.  

This pipeline enables users to tailor bounding box dimensions selectively, offering flexibility to refine and trim the dataset. Users can adjust dimensions to focus on areas of interest and verify whether all critical elements (connectors, pre-synaptic neurons, and post-synaptic neurons) are contained within the defined bounding boxes.  

### **3D visualization of the bounding box and their respective synapse**
After defining the bounding box dimensions, a 3D plot and `df` examines points within the bounding box to ensure all relevant neuron data—connectors, pre-, and postsynaptic coordinates—are included in the analysis.
  -	A subset of data, `inside_box`, is created for points meeting all three conditions: connector, pre-, and post-coordinates fall inside the bounding box.
  -	Data not satisfying these conditions is excluded and categorized into separate lists: `pre_outside_box` and `post_outside_box`, which capture points where only pre- or post-coordinates fall outside the bounding box.
The bounding box dimensions can be dynamically expanded or contracted to include or exclude specific synapses, ensuring optimal representation of the dataset for training.
Finally, the processed data is visualized:
  -	A 3D scatter plot displays points inside and outside the bounding box after adjustments.
  -	A wireframe representation of the bounding box is included as a reference for spatial context.
  


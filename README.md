# synapse_CATMAID
In Drosophila, Mushroom Body Output Neurons (MBONs) are the primary output of the mushroom body (MB), the brain’s key centre for higher cognitive processing. MBONs integrate valence signals from dopaminergic neurons (DANs), enabling memory-guided behaviours such as odour-based decision-making. Synapses within MBONs thus provides a viable quantitative measure for associative learning conveyed by DANs and other MB intrinsic neurons, which encodes memories within the MBONs.
In this pipeline, our aim is to retrieve synapses of the MBONs in the j2 compartment to advance our machine learning model in detecting synapses. Ground-truth synapses and neuron skeleton of the MBON-j2 were traced with CATMAID on both hemispheres. Using pymaid, our analysis effectively addresses the challenge of retrieving specific pre- and post-synaptic partners of MBON-j2 and their locations within the volume. This information is crucial for enabling our model to learn the cues and features necessary to detect discrepancies between the presynaptic and postsynaptic neurons, along with their associated T-bars. Ultimately, a 3D bounding box from the MBON j2 compartment was cropped, where model detection was applied and ground-truth coordinates of the synapses from the box were retrieved to facilitate the training of the model. 

# Latent graph by CMP-LGI
The latent graph obtained after multiple discrete sampling based on the edge probability set learned by CMP-LGI.
Pyvis is a good visual graph structure tool (https://pyvis.readthedocs.io/en/latest/). In the application case of the CMP-LGI algorithm, pyvis is used to visualize the graph structure obtained by discrete sampling. Please use a browser to open 'CMP-LGI_black.html', such as Chrome, Firefox, Edge, etc. The nodes in the graph correspond to the indicators in 'https://github.com/With-the-sun/PN-CRAR-mode-s-casedata.git '

# Graph_Fourier_transform
The folder "CMP-LGI for github" is used to implement “Effective analysis of heterogeneous data” and requires a MATLAB version later than R2021b. It contains six files, among which Graph_Fourier_transform_SWAT.m is the primary script that can be executed directly:

Graph_Fourier_transform_SWAT.m: The main, directly runnable script.
SWaT_Dataset_Normal_v1.xlsx: The original dataset from SWaT used in this work.
Data.mat: A MATLAB-saved version of SWaT_Dataset_Normal_v1.xlsx, allowing users to operate on the data directly and avoid repeatedly loading the XLSX file.
A_P&ID.mat: An adjacency matrix derived from the piping and instrumentation diagram (P&ID), serving as a concrete representation of prior knowledge.
A_CMP-LGI: The graph adjacency matrix learned by the CMP-LGI algorithm.
CMP-LGI_Fourier Transform_directed_800dpi.tif: A visualization showcasing the results produced by running Graph_Fourier_transform_SWAT.m.

# Quantum-Machine-Learning-
Quantum Kernel-Enhanced Hybrid Models for Start-Up
Success Prediction
Abstract
Quantum Machine Learning (QML) offers a transformative approach to solving complex
classification problems by leveraging quantum computational principles. This paper explores
the use of quantum kernel methods—specifically Quantum Support Vector Machines
(QSVMs)—for classifying start-up companies as successful or unsuccessful based on venture
capital (VC) investment data. Building upon the foundational work “Quantum Machine
Learning: Quantum Kernel Methods” by Sanjeev Naguleswaran, we propose a novel hybrid
architecture that integrates quantum kernel preprocessing into classical Convolutional Neural
Networks (CNNs). Experimental results demonstrate that quantum-enhanced models can
achieve comparable or superior performance to classical methods, suggesting promising
future applications of QML in financial analytics.
1. Introduction and Background
Quantum Machine Learning (QML) offers a promising approach to solving complex machine
learning problems by embedding data into high-dimensional quantum feature spaces. Among
the various QML techniques, quantum kernel methods stand out for their ability to potentially
achieve quantum advantage in pattern recognition tasks.
The venture capital (VC) industry traditionally relies on human intuition and manual
evaluations to predict the success of start-ups, a method that is often time-consuming and
subjective. Although classical machine learning (ML) models offer partial automation, their
representational power is limited by classical computation. Quantum kernel methods, by
contrast, allow data to be embedded into complex Hilbert spaces, providing a richer feature
space that could improve classification accuracy.
This research focuses on applying QSVMs to classify start-up companies based on VC
investment data and extends the idea by integrating quantum preprocessing into a classical
CNN framework, aiming for enhanced performance.
2. Related Work
Previous research, notably by Sanjeev Naguleswaran in “Quantum Machine Learning:
Quantum Kernel Methods”, demonstrated the potential of quantum kernels for classification
tasks on synthetic and small datasets. Classical machine learning models such as Decision
Trees, Random Forests, and SVMs have also been extensively used in financial and
investment analytics, but often fall short when dealing with complex, high-dimensional, and
noisy data.
Hybrid quantum-classical models have recently emerged as a promising approach, with
preliminary studies suggesting that quantum feature mappings can enhance classical learning
algorithms, especially in the context of deep learning. However, few studies have tested these
models on large, noisy, real-world datasets like VC investment data.
3. Proposed Work
This research introduces a comprehensive methodology for evaluating the effectiveness of
quantum-enhanced models in real-world financial analytics:
3.1 Dataset
We utilize Crunchbase data consisting of 49,437 start-up companies. After filtering for only
“exited” (successful) and “closed” (unsuccessful) companies, 5,497 entries remained. Feature
engineering refined the dataset to 17 meaningful features, such as founding date, investment
type, funding stages, and geographic location. The dataset’s inherent noisiness offers a
realistic testbed for QML applications.
3.2 Methodology Overview
• Classical Benchmarking:
We first trained standard classifiers, including Decision Trees, Random Forests, and
classical SVMs. The best classical models achieved accuracies of up to 68%.
• Quantum Kernel SVM (QSVM):
Using Pennylane, we developed a quantum kernel and substituted it into the classical
SVM framework. The QSVM achieved comparable performance (~66% accuracy),
suggesting that quantum feature mapping can match classical performance even at
small scales.
• Hybrid Quantum-CNN Model:
A novel hybrid model was designed, introducing a quantum kernel-based
convolutional layer as a preprocessing step before the classical CNN. This hybrid
approach showed improvements over the standalone CNN in both accuracy and loss
metrics, indicating that quantum feature mapping enhanced the CNN's
representational power.
3.3 Key Contributions
• Demonstrating the feasibility of QSVMs on real-world VC data.
• Introducing a new Quantum Kernel + CNN hybrid architecture.
• Providing empirical evidence that quantum enhancement can yield tangible
performance gains even with noisy, real-world datasets.
4. Results
5. Conclusion
This research validates that quantum kernel methods can match, and in some cases surpass,
classical machine learning techniques in classifying start-up success based on VC investment
data. By integrating quantum feature mappings with classical CNN architectures, we
achieved improved learning outcomes, highlighting a tangible quantum advantage.
Our findings open new pathways for applying Quantum Machine Learning in finance,
particularly for venture capital firms seeking to optimize investment decisions with limited
resources. Future work will explore optimizing quantum circuit designs and scaling hybrid
models to larger datasets and multi-class investment scenarios

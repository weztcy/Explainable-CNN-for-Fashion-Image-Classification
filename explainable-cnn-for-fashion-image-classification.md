# Portfolio Project Data — Explainable CNN for Fashion Image Classification

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/explainable_cnn_for_fashion/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image12.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "explainable-cnn-for-fashion-image-classification",
  "code": "PRO-40 // ML-25",
  "title": "Explainable CNN for Fashion Image Classification",
  "category": "machine-learning",
  "categoryLabel": "Deep Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Explainable-CNN-for-Fashion-Image-Classification",
  "license": "MIT License",
  "banner": "/images/projects/explainable_cnn_for_fashion/banner.webp",
  "summary": "Comparative PyTorch CNN pipeline achieving 92.61% test accuracy on Fashion-MNIST with Grad-CAM attribution and interactive Streamlit diagnostics.",
  "projectDescription": "An explainable PyTorch vision pipeline benchmarking MLP, Standard CNN, and Regularized CNN models on 70,000 Fashion-MNIST images. Achieves 92.61% test accuracy and integrates Grad-CAM spatial attribution on conv4 with an interactive dark-mode Streamlit dashboard.",
  "architectureSummary": "IDX Audit & Stratified Split -> Train-Only Normalization & Spatial Augmentation -> Regularized CNN (conv4) -> Test Evaluation & Grad-CAM -> Streamlit UI",
  "techStack": [
    "Python",
    "PyTorch",
    "Torchvision",
    "NumPy",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "Pillow",
    "Streamlit",
    "PyTest"
  ],
  "telemetryMatrix": {
    "projectType": "Explainable Vision Classifier",
    "targetPlatform": "CPU Inference & Streamlit Web App",
    "primaryDomain": "Computer Vision & Apparel",
    "architecturePipeline": "Regularized CNN + Grad-CAM (conv4)",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Explainable Image Classification Pipeline",
    "learningParadigm": "Deep Learning",
    "modelDomain": "Apparel Vision Classification",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Regularized Dual-Conv CNN",
      "Grad-CAM Attribution",
      "Interactive Streamlit Diagnostics"
    ]
  },
  "mlProblemContext": "Classifying 28x28 grayscale Fashion-MNIST apparel suffers from severe fine-grained overlap across upper-body garments such as Shirt, T-shirt/top, Pullover, and Coat. Furthermore, standard CNNs act as opaque black boxes without exposing which garment contours drive predictions or trigger class confusion.",
  "mlDatasetOverview": {
    "profileBadge": "Fashion-MNIST Benchmark",
    "source": "Zalando Research Fashion-MNIST (dataset/original/*.gz)",
    "modality": "Computer Vision (Grayscale)",
    "scale": "70,000 Images (54k Train / 6k Val / 10k Test)",
    "inputShape": "N x 1 x 28 x 28 Float32 Tensor",
    "dataFormat": "IDX Binary Gzip (.gz)",
    "description": "Balanced 10-class apparel dataset verified via official MD5 checksums with zero empty frames and zero train-test duplicate overlap. Stratified 90:10 splitting (seed=42) yields 5,400 train and 600 validation samples per class while isolating 10,000 test images.",
    "provenanceNote": "Official Zalando Research IDX binary archives verified programmatically against canonical MD5 hashes."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Spatial Tensor & 10-Class Space",
    "spaceNote": "28x28 grayscale intensities centered on dark backgrounds and standardized via train-only statistics.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Fashion Apparel Category",
      "type": "Multi-Class (10 Categories)",
      "distribution": "Exact uniform balance: 5,400 train / 600 val / 1,000 test images per class"
    },
    "inputSignals": [
      {
        "name": "Normalized Grayscale Spatial Grid",
        "type": "Spatial Tensor (1 x 28 x 28)",
        "importanceOrRole": "Primary Structural Input",
        "description": "Pixel intensities scaled from [0, 255] to [0, 1] and standardized with train-only mean (0.2859) and std (0.3530)."
      },
      {
        "name": "Spatial Augmentation Perturbations",
        "type": "Train-Only Regularizer",
        "importanceOrRole": "Generalization Enhancer",
        "description": "Random horizontal flip (p=0.5) and 2-pixel reflection-padded random crop applied strictly during Regularized CNN training."
      },
      {
        "name": "Final Convolutional Activation Map (conv4)",
        "type": "Feature Tensor (64 x 7 x 7)",
        "importanceOrRole": "Grad-CAM Localization Target",
        "description": "High-level spatial activations weighted by backpropagated class gradients to localize collars, straps, handles, and soles."
      }
    ],
    "dataCharacteristicsSummary": "Training pixels exhibit a dominant zero-intensity background peak and a bimodal foreground distribution. Mean class heatmaps reveal strong silhouette separation for Trouser, Bag, and footwear, alongside heavy torso and sleeve overlap among Shirt, T-shirt/top, Pullover, and Coat."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Deep Learning for Multi-Class Visual Recognition & Spatial Attribution",
    "methodologyDescription": "Three PyTorch architectures—Baseline MLP, Standard CNN, and Regularized CNN—are trained under identical Cross-Entropy budgets and compared on validation loss to isolate the impact of 2D convolutions, Batch Normalization, Dropout, and spatial augmentation.",
    "justification": "Flattened MLPs discard 2D spatial locality, while unregularized CNNs overfit after epoch 6. Dual-conv blocks paired with BatchNorm, Dropout, and Grad-CAM on conv4 maximize generalization while keeping inference interpretable."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "IDX Binary Verification & Duplicate Audit",
      "description": "Validates magic numbers (2051/2049), official MD5 checksums, [0, 255] bounds, zero empty images, and 0 pixel-level duplicates between train and test.",
      "badge": "Integrity Audit"
    },
    {
      "step": "02",
      "title": "Deterministic Stratified Train/Val Split",
      "description": "Partitions the 60,000 training images 90:10 with seed=42 into 54,000 train and 6,000 validation records, persisting indices to split_indices.json.",
      "badge": "Anti-Leakage"
    },
    {
      "step": "03",
      "title": "Train-Only Normalization & Spatial Augmentation",
      "description": "Computes mean (0.2859) and std (0.3530) exclusively on the 54,000 training images and applies horizontal flip plus reflection-pad cropping only during training.",
      "badge": "Tensor Scaling"
    }
  ],
  "mlDataPreparationNote": "Normalization statistics and spatial augmentations are strictly restricted from validation and test loaders; the 10,000-sample test split remains isolated until final evaluation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "RegularizedCNN (Dual-Conv + BatchNorm + Dropout)",
    "architectureSummary": "Two dual-conv blocks (1->32->32 and 32->64->64, 3x3 kernels, BatchNorm2d, ReLU, MaxPool2d 2x2, Dropout2d 0.25) feeding a 3,136->256->10 classifier head with BatchNorm1d and Dropout(0.4), totaling 871,082 parameters.",
    "configurationBadge": "871,082 Parameters",
    "hyperparameters": [
      {
        "name": "batch_size",
        "value": "128",
        "description": "Mini-batch size across training and validation epochs."
      },
      {
        "name": "learning_rate",
        "value": "0.001",
        "description": "Initial Adam step size decayed by 0.5x via ReduceLROnPlateau (patience=2)."
      },
      {
        "name": "weight_decay",
        "value": "1e-4",
        "description": "L2 weight regularization penalty applied in Adam."
      },
      {
        "name": "dropout_rate",
        "value": "0.25 / 0.40",
        "description": "2D spatial dropout (0.25) after conv pools and dense dropout (0.40) before output logits."
      },
      {
        "name": "early_stopping_patience",
        "value": "3 epochs",
        "description": "Halts training if validation loss fails to improve over 3 consecutive epochs (max 12)."
      }
    ],
    "objectiveCriterion": "Categorical Cross-Entropy Loss",
    "optimizationSolver": "Adam with ReduceLROnPlateau",
    "trainingProcess": "Trained for up to 12 epochs on CPU (528.4s total) with deterministic seed 42 and validation-loss checkpointing. Regularized CNN steadily lowered validation loss from 0.3131 to a minimum of 0.1842 at epoch 11, avoiding the validation oscillation observed in Standard CNN."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Stratified Validation Selection + Single-Pass Isolated Test Evaluation",
    "description": "Checkpoint selection uses 6,000 stratified validation samples, followed by a single isolated evaluation pass over 10,000 unseen test images covering accuracy, macro/weighted F1, confusion matrices, and 100-run CPU latency benchmarks.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "92.61%",
        "benchmark": "Baseline > 88.73%"
      },
      {
        "label": "TEST MACRO F1",
        "value": "92.58%",
        "benchmark": "Baseline > 88.75%"
      },
      {
        "label": "BEST VAL LOSS",
        "value": "0.1842",
        "benchmark": "Baseline < 0.2118"
      },
      {
        "label": "CPU LATENCY (B=1)",
        "value": "0.71 ms",
        "benchmark": "Target < 5.0 ms"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Tab 1 — Live Sample Classification & Confidence",
      "badge": "Interactive UI",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image01.webp",
      "telemetryNote": "Displays test sample #0000 classified as Ankle boot with 99.86% softmax confidence."
    },
    {
      "title": "Streamlit Tab 1 — Grad-CAM Overlay & 10-Class Probabilities",
      "badge": "Explainability UI",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image02.webp",
      "telemetryNote": "Visualizes conv4 Grad-CAM overlay alongside 99.9% softmax probability bar chart."
    },
    {
      "title": "Streamlit Tab 2 — Test Evaluation & KPI Dashboard",
      "badge": "Metrics Dashboard",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image03.webp",
      "telemetryNote": "Surfaces 92.61% test accuracy, 92.58% Macro F1, 871K params, and comparison plots."
    },
    {
      "title": "Streamlit Tab 3 — Dataset Specs & CNN Layer Architecture",
      "badge": "Architecture View",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image04.webp",
      "telemetryNote": "Documents train-only normalization stats, class balance, and RegularizedCNN layers."
    },
    {
      "title": "Fashion-MNIST Gallery & Grad-CAM Attribution Heatmaps",
      "badge": "Grad-CAM Gallery",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image05.webp",
      "telemetryNote": "Pairs 10-class sample grid with conv4 Grad-CAM overlays for correct and confused cases."
    },
    {
      "title": "Test Prediction Gallery (Correct vs. Misclassified)",
      "badge": "Error Gallery",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image06.webp",
      "telemetryNote": "Contrasts high-confidence Trouser/Pullover/Boot predictions against Shirt confusions."
    },
    {
      "title": "Pixel Intensity Histogram & Class-Wise Mean Heatmaps",
      "badge": "Exploratory Data",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image07.webp",
      "telemetryNote": "Shows dominant zero background peak and upper-body silhouette overlap across 54k images."
    },
    {
      "title": "Training & Validation Loss and Accuracy Curves",
      "badge": "Convergence Curves",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image08.webp",
      "telemetryNote": "Regularized CNN reaches 0.1842 val loss at epoch 11 without Standard CNN overfitting."
    },
    {
      "title": "Multi-Model Performance & Parameter Complexity Comparison",
      "badge": "Model Benchmark",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image09.webp",
      "telemetryNote": "Compares MLP (88.73%), Standard CNN (91.68%), and Regularized CNN (92.61%) test accuracy."
    },
    {
      "title": "Stratified Split Class Distribution (54k / 6k / 10k)",
      "badge": "Split Balance",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image10.webp",
      "telemetryNote": "Verifies exact 5,400 train, 600 val, and 1,000 test samples per category (seed=42)."
    },
    {
      "title": "Per-Class Precision, Recall & F1-Score Breakdown",
      "badge": "Class Metrics",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image11.webp",
      "telemetryNote": "Highlights Trouser (99.25% F1) and Sandal (98.54% F1) versus Shirt (77.18% F1)."
    },
    {
      "title": "Test Confusion Matrices (Raw Counts & Normalized Recall)",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/explainable_cnn_for_fashion/image12.webp",
      "telemetryNote": "Isolates 739 errors primarily to Shirt, T-shirt/top, Coat, and Pullover pairs."
    }
  ],
  "mlVisualizationTelemetryNote": "Regularized CNN maintains tight alignment between training (91.52% at epoch 11) and validation (93.27%) curves, generalizing cleanly to 92.61% on the isolated test set.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Grad-CAM gradients from conv4 show that correct predictions lock onto class-defining topology such as trouser leg bifurcation, sandal strap gaps, bag handles, and shoe contours. Conversely, misclassified upper-body garments exhibit diffuse mid-torso activation where 28x28 resolution obscures collars and buttons.",
    "keyDrivers": [
      {
        "factor": "2D Spatial Inductive Bias & Dual-Conv Blocks",
        "impact": "+3.88% test accuracy gain over flattened Baseline MLP (88.73% to 92.61%)."
      },
      {
        "factor": "BatchNorm, Dropout & Reflection Augmentation",
        "impact": "Lowered best validation loss from 0.2118 to 0.1842 and eliminated Standard CNN overfitting."
      },
      {
        "factor": "Distinct Silhouette Morphology (Trouser/Sandal/Bag)",
        "impact": "Achieves 98.36% to 99.25% F1 with zero cross-domain errors against footwear or trousers."
      },
      {
        "factor": "Upper-Body Silhouette Overlap (Shirt/T-shirt/Coat)",
        "impact": "Accounts for top confusion pairs (Shirt->T-shirt/top: 92; T-shirt/top->Shirt: 86; Shirt->Coat: 71)."
      }
    ],
    "limitations": [
      {
        "title": "Low 28x28 Grayscale Resolution",
        "description": "Centered monochrome Fashion-MNIST silhouettes erase collars, seams, and fabric textures needed to separate Shirt from T-shirt/top and Coat, and do not reflect unconstrained e-commerce photos."
      },
      {
        "title": "Coarse 7x7 Grad-CAM Spatial Grid",
        "description": "Target layer conv4 outputs 7x7 activation maps prior to bilinear upsampling, yielding diffuse heatmaps that reflect local gradient correlation rather than causal reasoning."
      },
      {
        "title": "CPU-Bounded Training Scope",
        "description": "Experiments were executed on CPU without GPU acceleration, restricting exploration of heavier pretrained backbones like ResNet-50 or Vision Transformers."
      }
    ],
    "futureImprovements": [
      {
        "title": "High-Resolution In-the-Wild Transfer Learning",
        "target": "Fine-tune ConvNeXt or Swin Transformer backbones on 256x256 RGB apparel datasets such as DeepFashion."
      },
      {
        "title": "Comparative Axiom-Based Attribution",
        "target": "Benchmark Grad-CAM against Integrated Gradients, Score-CAM, and KernelSHAP for attribution stability."
      },
      {
        "title": "Margin-Aware Loss Regularization",
        "target": "Evaluate Focal Loss or supervised contrastive objectives to sharpen separation between Shirt, T-shirt/top, and Coat."
      }
    ],
    "discoveryNote": "Grad-CAM attribution is extracted directly from RegularizedCNN.conv4 via PyTorch forward/backward hooks and bilinearly upsampled to 28x28."
  },
  "mlModelArtifacts": [
    {
      "name": "models/best_cnn.pt",
      "format": "PT"
    },
    {
      "name": "models/regularized_cnn.pt",
      "format": "PT"
    },
    {
      "name": "models/standard_cnn.pt",
      "format": "PT"
    },
    {
      "name": "models/mlp.pt",
      "format": "PT"
    },
    {
      "name": "metrics.json",
      "format": "JSON"
    },
    {
      "name": "split_indices.json",
      "format": "JSON"
    },
    {
      "name": "run_config.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Deep Learning",
    "modelArchitecture": "Regularized Dual-Conv CNN",
    "coreFramework": "PyTorch",
    "validationMethod": "Stratified Split & Test"
  }
}
```

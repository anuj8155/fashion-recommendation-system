# Fashion Recommendation System

A deep learning-based fashion recommendation system that uses ResNet50 for feature extraction and k-nearest neighbors algorithm to find similar fashion items.

## 🎯 Overview

This system allows users to upload a fashion item image and get recommendations for similar items from a pre-processed dataset. It uses transfer learning with ResNet50 to extract meaningful features from fashion images and employs cosine similarity to find the most similar items.

## 📁 Project Structure

```
fashion-recommender/
│
├── 📄 app.py                    # Feature extraction and preprocessing script
│   └── → Processes entire dataset
│   └── → Generates embeddings.pkl & filenames.pkl
│
├── 🌐 main.py                   # Streamlit web application
│   └── → User interface for recommendations
│   └── → Handles file uploads and displays results
│
├── 🧪 test.py                   # Testing script with OpenCV visualization
│   └── → Tests system with sample images
│   └── → Displays results using OpenCV
│
├── 📊 Generated Files:
│   ├── embeddings.pkl           # Serialized feature vectors
│   └── filenames.pkl            # Serialized image file paths
│
├── 📁 Directories:
│   ├── images/                  # Fashion dataset images
│   │   ├── item_001.jpg
│   │   ├── item_002.jpg
│   │   └── ... (dataset images)
│   │
│   ├── uploads/                 # User uploaded images
│   │   └── (temporary storage)
│   │
│   └── sample/                  # Sample images for testing
│       └── shirt.jpg
│
└── 📋 README.md                 # This documentation file
```

## 🔧 Dependencies

```
pip install tensorflow
pip install streamlit
pip install scikit-learn
pip install opencv-python
pip install pillow
pip install numpy
pip install tqdm
```

## 📋 File Descriptions

### 1. app.py - Feature Extraction Pipeline

**Purpose**: Preprocesses the entire fashion dataset and extracts feature vectors.

**Key Components**:
- **Model Setup**: Uses ResNet50 pre-trained on ImageNet, removes the top classification layer
- **Feature Extraction Function**: 
  - Loads and resizes images to 224x224 pixels
  - Applies ResNet50 preprocessing
  - Extracts features using GlobalMaxPooling2D
  - Normalizes feature vectors using L2 normalization
- **Batch Processing**: Processes all images in the 'images' directory
- **Serialization**: Saves features and filenames as pickle files for later use

**Workflow**:
1. Initialize ResNet50 model without top layer
2. Add GlobalMaxPooling2D layer for feature extraction
3. Iterate through all images in 'images' directory
4. Extract normalized feature vectors for each image
5. Save features and filenames as pickle files

### 2. main.py - Streamlit Web Application

**Purpose**: Provides a user-friendly web interface for the recommendation system.

**Key Components**:
- **File Upload**: Streamlit file uploader for user images
- **Feature Extraction**: Same process as app.py but for single uploaded image
- **Recommendation Engine**: Uses k-nearest neighbors to find similar items
- **Results Display**: Shows 5 most similar fashion items in columns

**User Flow**:
1. User uploads a fashion item image
2. System saves the uploaded file
3. Feature extraction is performed on the uploaded image
4. k-NN algorithm finds 5 most similar items
5. Results are displayed in a grid layout

### 3. test.py - Testing and Validation Script

**Purpose**: Tests the recommendation system with a sample image and displays results.

**Key Components**:
- **Model Loading**: Same ResNet50 setup as other files
- **Sample Processing**: Processes 'sample/shirt.jpg'
- **Similarity Search**: Finds 6 nearest neighbors (including the query image)
- **Visualization**: Uses OpenCV to display similar images

**Testing Process**:
1. Load pre-computed features and filenames
2. Process the sample image
3. Find nearest neighbors
4. Display results using OpenCV windows

## 🔄 System Workflow

### Phase 1: Dataset Preprocessing (app.py)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │───▶│   ResNet50       │───▶│   Feature       │
│   Images        │    │   Model          │    │   Extraction    │
│   (images/)     │    │   Processing     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Save          │◀───│   L2             │◀───│   Global Max    │
│   embeddings.pkl│    │   Normalization  │    │   Pooling       │
│   filenames.pkl │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Phase 2: Web Application Flow (main.py)
```
┌─────────────────┐
│   User Opens    │
│   Streamlit App │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Upload        │───▶│   Save to        │───▶│   Extract       │
│   Fashion Image │    │   uploads/       │    │   Features      │
└─────────────────┘    └──────────────────┘    └─────────┬───────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Display 5     │◀───│   k-NN Algorithm │◀───│   Load Pre-     │
│   Similar Items │    │   Find Neighbors │    │   computed      │
│   in Grid       │    │                  │    │   Features      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Phase 3: Testing Flow (test.py)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sample Image  │───▶│   Feature        │───▶│   Load          │
│   (shirt.jpg)   │    │   Extraction     │    │   embeddings.pkl│
└─────────────────┘    └──────────────────┘    └─────────┬───────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenCV        │◀───│   Find 6 Nearest │◀───│   k-NN Search   │
│   Display       │    │   Neighbors      │    │   Algorithm     │
│   Results       │    │   (inc. query)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Complete System Architecture
```
                    ┌─────────────────────────────────────────┐
                    │           PREPROCESSING PHASE           │
                    └─────────────────┬───────────────────────┘
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                                                                   │
    │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────┐ │
    │  │   Images    │─▶│   ResNet50   │─▶│ GlobalMax   │─▶│   L2    │ │
    │  │ Directory   │  │   Feature    │  │ Pooling     │  │  Norm   │ │
    │  │             │  │ Extraction   │  │             │  │         │ │
    │  └─────────────┘  └──────────────┘  └─────────────┘  └─────────┘ │
    │                                                                   │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  embeddings.pkl │
                    │  filenames.pkl  │
                    └─────────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  STREAMLIT    │    │     TESTING     │    │    FUTURE       │
│  WEB APP      │    │     SCRIPT      │    │  INTEGRATIONS   │
│               │    │                 │    │                 │
│ ┌───────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │User Upload│ │    │ │Sample Image │ │    │ │   REST API  │ │
│ │           │ │    │ │   Testing   │ │    │ │             │ │
│ └─────┬─────┘ │    │ └─────┬───────┘ │    │ └─────────────┘ │
│       │       │    │       │         │    │                 │
│       ▼       │    │       ▼         │    │ ┌─────────────┐ │
│ ┌───────────┐ │    │ ┌─────────────┐ │    │ │  Database   │ │
│ │k-NN Search│ │    │ │OpenCV Display│ │    │ │ Integration │ │
│ │           │ │    │ │             │ │    │ │             │ │
│ └─────┬─────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│       │       │    │                 │    │                 │
│       ▼       │    │                 │    │                 │
│ ┌───────────┐ │    │                 │    │                 │
│ │Grid Display│ │    │                 │    │                 │
│ │Results    │ │    │                 │    │                 │
│ └───────────┘ │    │                 │    │                 │
└───────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Setup and Usage

### Step 1: Prepare Dataset
1. Create an 'images' directory
2. Add your fashion dataset images to this directory
3. Create 'uploads' and 'sample' directories

### Step 2: Extract Features
```bash
python app.py
```
This will:
- Process all images in the 'images' directory
- Generate 'embeddings.pkl' and 'filenames.pkl' files
- May take time depending on dataset size

### Step 3: Run the Web Application
```bash
streamlit run main.py
```
This will:
- Start the Streamlit web server
- Open the application in your browser
- Allow you to upload images and get recommendations

### Step 4: Test the System (Optional)
```bash
python test.py
```
This will:
- Test the system with the sample image
- Display results using OpenCV windows
- Press any key to cycle through similar images

## 🧠 Technical Details

### Feature Extraction Process
1. **Image Preprocessing**: Resize to 224x224, convert to array, expand dimensions
2. **ResNet50 Processing**: Extract deep features using pre-trained weights
3. **Global Max Pooling**: Reduce spatial dimensions while preserving important features
4. **L2 Normalization**: Normalize feature vectors for cosine similarity

### Similarity Calculation
- Uses Euclidean distance in the normalized feature space
- k-NN algorithm with brute force search for accuracy
- Returns top 5 most similar items (excluding the query image)

### Model Architecture
```
Input Image (224x224x3)
↓
ResNet50 (without top layer)
↓
GlobalMaxPooling2D
↓
Feature Vector (2048 dimensions)
↓
L2 Normalization
↓
Similarity Search
```

## 📊 Performance Considerations

- **Feature Extraction**: One-time process, can be computationally intensive
- **Search Speed**: Fast k-NN search after preprocessing
- **Memory Usage**: Stores all feature vectors in memory
- **Scalability**: Consider using approximate nearest neighbors (ANN) for large datasets

## 🔧 Customization Options

1. **Change Model**: Replace ResNet50 with other pre-trained models (VGG, EfficientNet)
2. **Adjust Neighbors**: Modify n_neighbors parameter for more/fewer recommendations
3. **Different Metrics**: Use cosine similarity instead of Euclidean distance
4. **UI Improvements**: Enhance Streamlit interface with additional features

## 🐛 Troubleshooting

### Common Issues:
1. **Missing pickle files**: Run app.py first to generate feature files
2. **Memory errors**: Reduce batch size or use smaller model
3. **Streamlit beta_columns error**: Update to st.columns() for newer Streamlit versions
4. **OpenCV display issues**: Ensure proper OpenCV installation and display environment

### File Path Issues:
- Ensure 'images', 'uploads', and 'sample' directories exist
- Check file permissions for reading/writing pickle files
- Verify image file formats are supported (jpg, png, etc.)

## 📈 Future Enhancements

1. **Database Integration**: Store features in a database instead of pickle files
2. **Real-time Processing**: Implement streaming feature extraction
3. **Advanced Filtering**: Add category, color, and price filters
4. **User Feedback**: Implement rating system for recommendation improvement
5. **API Development**: Create REST API for integration with other applications

## 📄 License

This project is open source and available under the MIT License.

# 🌿 Leaf Lens - Plant Disease Detection System 

Leaf Lens is an Edge AI solution created during Qualcomm's Edge AI Developer Hackathon-2025, delivering offline plant disease detection for farmers. Our cross-functional team built an optimized CNN model that identifies diseases in tomato, potato and pepper crops with *93% accuracy*, providing instant multilingual treatment advice without internet dependency.

### 🌟 Key Features

- Accurate Disease Detection : CNN model identifies 10+ diseases with 95% accuracy
- Complete Offline Operation : Works without internet after initial setup
- Multilingual Support : English, Hindi
- Comprehensive Reports:
  - Disease identification with confidence score
  - Symptoms description
  - Treatment protocols
  - Preventive measures
  - Fertilizer recommendations
  - Yield impact analysis
- LLM-Powered Insights: LLaMA/Ollama provides detailed agricultural advice


### 📊 Dataset & Model

####  Dataset

```bash
  https://www.kaggle.com/datasets/moazeldsokyx/plantvillage
   ```
####  Model
```bash
  https://www.kaggle.com/code/moazeldsokyx/plant-leaf-diseases-detection-using-cnn
   ```

###  🌿Plants & Diseases Covered:
| Plant     |  Diseases | Example Diseases                |
|------------|---------------|---------------------------------------|
| Tomato  | 12             | Early Blight, Late Blight        |
| Potato   | 8               | Potato Scab, Wilt                  |
| Pepper  | 5               | Bacterial Spot, Anthracnose |

###  📊Model Performance:
- Test Accuracy: 93.2%
- Inference Time: 0.8s (CPU)
- Model Size: 28MB (quantized)


### 🛠️ Tech Stack

#### ⚙️Core Components:
- **Frontend** : Next.js (offline capable)
- **Backend** : Flask API
- **AI Model** : Custom CNN (TensorFlow/Keras)
- Optimized with **SNPE (Snapdragon Neural Processing Engine)**
  - Quantized using **Qualcomm QNN (Qualcomm Neural Networks SDK)**
  - TTS using **Helsinki-NLP/opus-mt-en-hi**
- **LLM Integration** : LLaMA/Ollama (AnythingLLM)
- **Edge Deployment** : Snapdragon-powered devices
- **LLM Integration** : LLaMA/Ollama (AnythingLLM)

### 🚀 Installation

 Prerequisites
- Python 3.8+
- Node.js 16+
- Ollama (for LLM features)

### Setup Instructions

#### 1. Clone Repository:
   ```bash
   git clone https://github.com/Balajisix/Leaflens-Qualcomm-Final.git
   cd leaf-lens
   ```

#### 2. Backend Setup:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

#### 3. Frontend Setup:
   ```bash
   cd ../frontend
   npm install
   ```

#### 4. Anything LLM Setup:
   Create a config.yaml file in your project structure
   ```bash
   api-key: your_api_key
   workspace_slug: your_slug
   stream: true
   stream_timeout: 60
   ```


### 🖥️ Usage

#### 1. Start Backend:
   ```bash
   cd leaflens-backend
   python app.py
   ```

#### 2. Start Frontend:
   ```bash
   cd leaflens-frontend
   npm run dev
   ```

#### 3. Access Application:
   - Open `http://localhost:3000` in browser
   - Upload plant leaf image
   - View detailed disease report

### Sample Workflow :
#### 1. User uploads tomato leaf image
#### 2. System detects "Early Blight" (92% confidence)
#### 3. Displays :
   - Symptoms: "Small brown spots with concentric rings"
   - Treatment: "Apply copper fungicide weekly"
   - Prevention: "Rotate crops annually"
   - Fertilizer: "Balanced NPK (10-10-10)"
#### 4. Text to speech translation


### 📈 Performance Metrics

#### Model Evaluation :
| Metric        | Score  |
|--------------- |----------|
| Accuracy   | 93.2% |
| Precision   | 91.8% |
| Recall        | 92.5% |
| F1-Score   | 92.1% |

### ⚙️ Hardware Requirements 

- Processor	Snapdragon X Elite (includes Hexagon NPU for on-device AI)
      - NPU (Neural Processing Unit)	Integrated Hexagon NPU in Snapdragon X Elite (no external NPU required)
- RAM	8 GB minimum, 16 GB or higher recommended for larger models
- Storage	20 GB+ free SSD space for SDKs, models and tools
- GPU	Not required for NPU inference
- Adreno GPU present but optional for AI tasks
- Battery & Thermals	Efficient cooling recommended, as NPU workloads can generate heat

### 👥 Project Team

#### 💻 Team Members

🧑🚀 Balaji V     
📩 balajivs0305@gmail.com   
🔗 https://www.linkedin.com/in/balaji-v-544984215    
🖥️ *Software Engineer*     
🔍 https://balaji-portfolio-website.vercel.app/                      

👩🔧 Nivethithaa Siva <br>
📩 nivethithaasiva@gmail.com <br>
🔗 https://www.linkedin.com/in/nivethithaa-siva-3309b4249 <br>
🔁 *Data Pipeline Specialist*  
🔍 https://nivethithaa-portfolio.vercel.app/                         

🧑🌾 Bejoy JBT     
📩 bejoyjbt7@gmail.com    
🔗 https://www.linkedin.com/in/bejoyjbt    
🌐 *Edge AI Engineer*    
🔍 https://bejoy-portfolio.vercel.app/                                 

👩💻 Shuki Ravichandran    
📩 shukiravi03@gmail.com     
🔗 https://www.linkedin.com/in/shuki-ravichandran-383a78363    
🌱 *Agricultural Domain Expert*                               

### Repository:  
https://github.com//Balajisix/Leaflens-Qualcomm-Final


## 📜 License

MIT License - See [LICENSE](LICENSE) for details.
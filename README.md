# VerifyX – Deepfake Detection and Cybersecurity System

## 1. Project Title
**VerifyX – Media Authenticity Checker with Vector Memory**

---

## 2. Short Description
VerifyX checks whether an image or video is real or AI-generated.  
It uses deepfake detection models, Qdrant vector memory, and a simple explanation engine.  
The platform also includes a basic image generator and a defensive cybersecurity assistant.

---

## 3. Professional Overview
VerifyX helps users verify digital media with clarity and context.  
The system runs an ensemble deepfake model, converts the outputs into vectors, searches similar past cases in Qdrant, assigns a risk score, and provides a simple explanation.  
The frontend includes a left panel (image generation), right panel (secure detect), and a central chat interface.

---

## 4. Introduction
AI-generated images and videos are becoming hard to identify.  
VerifyX solves this problem by combining:

- Deepfake detection (images + videos)
- Memory using Qdrant vector search
- Simple natural-language explanations
- Risk scoring
- Frame-based video analysis
- A safe cybersecurity agent

**Main Purpose:**  
Help users understand whether a file is real or fake, and explain the result clearly.

---

# 5. Efficient Use of Qdrant (Memory Over Models)

VerifyX uses Qdrant as the memory core of the system.  
Each processed media file is converted into a **2-value vector**:

`[fake_logit, real_logit]`

### 5.1 Vector Storage
Each stored case includes:
- Vector  
- Label (real/fake)  
- Probabilities  
- Risk level  
- Media type  
- Frames used (videos)

### 5.2 Similarity Search
When a new file is checked, Qdrant returns the closest stored cases.  
This adds context and makes the decision more reliable.

### 5.3 Explainability
The system includes the retrieved cases in the final explanation.

### 5.4 Lightweight Design
Using 2-dimensional vectors makes the memory fast, simple, and scalable.  
This matches the “Memory Over Models” theme.

---

# 6. Visual Representation

## 6.1 Demo Video
(To be added)

## 6.2 System Workflow Diagram


                 +---------------------------+
                 |        User Upload        |
                 |  (Image / Video / Text)   |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 |       Preprocessing       |
                 | (PIL Conversion, Frames,  |
                 |   File Handling, Checks)  |
                 +-------------+-------------+
                               |
                               v
           +------------------------------------------------+
           |           Ensemble Deepfake Models             |
           |   (Model 1 + Model 2 + Model 3 → Probabilities)|
           +-------------------+----------------------------+
                               |
                               v
                 +---------------------------+
                 |     2D Vector Creation    |
                 |     [fake_logit, real]    |
                 +-------------+-------------+
                               |
                               v
        +------------------------------------------------------+
        |                     Qdrant Search                     |
        |  - Store vector                                      |
        |  - Retrieve similar cases (kNN)                      |
        |  - Return past results + metadata                    |
        +------------------------+-----------------------------+
                                 |
                                 v
        +------------------------------------------------------+
        |             Risk Engine + Explanation                |
        |  - Risk Level (Low / Medium / High)                  |
        |  - LLM Explanation (simple language)                 |
        |  - Uses similar retrieved cases                      |
        +------------------------+-----------------------------+
                                 |
                                 v
                 +---------------------------+
                 |        Final Output       |
                 | - Real/Fake Result        |
                 | - Probabilities           |
                 | - Risk Score              |
                 | - Explanation             |
                 +---------------------------+






## 6.3 Frontend Workflows

### Home Chat
- Simple agent chat  
- Smooth scrolling

### Left Panel: Image Generation
- Auto command: `#generate image`
- Submit prompt → generated image
- Options: view, download, copy link

### Right Panel: Secure Detect
- Upload file  
- Auto command: `#evaluation`
- Analysis saved as history card  
- Explanation shown in the main chat  

---

# 7. How to Use VerifyX

## 7.1 General Chat## 6.3 Frontend Workf
Type any message in the input bar and press Enter.

## 7.2 Image Generation
1. Open the left panel.  
2. Enter a prompt after the auto-filled command.  
3. Submit to generate an image.

## 7.3 Secure Detect
1. Open the right panel.  
2. Upload PNG/JPG/MP4.  
3. Enter prompt after the auto-filled command.  
4. Submit to see the result and explanation.

## 7.4 Cybersecurity Assistant
Ask any defensive cybersecurity question directly in chat.

---

# 8. Installation Guide

## 8.1 Clone the Repository

```bash
git clone <repository_url>
cd <project_folder>```


## 8.2 Install Dependencies

```bash
pip install -r requirements.txt

Copy code

---


## 8.3 Environment Variables
Create a `.env` file:



## 8.4 Run the Application
```bash
python app.py

Copy code

---



---

# 9. Team Contributions

## 9.1 AI Lead
- Built image and video deepfake pipeline  
- Designed ensemble system  
- Created 2D vector format for Qdrant  
- Implemented Qdrant storage and search  
- Added risk scoring  
- Added explanation engine  
- Integrated everything into the cyber agent  

## 9.2 Backend Engineer
- Added OCR, Whisper, BLIP, and Stable Horde  
- Managed multimedia APIs  
- Linked backend to the frontend  
- Maintained routing and processing  

## 9.3 Frontend Developer
- Designed UI layout  
- Implemented left/right panels  
- Added history, staging, and transitions  
- Built glass-style interface with smooth actions  

---

# 10. Experiment Attempts and Failures

## 10.1 Major Failures
- Models giving reversed real/fake outputs  
- Removed or broken Hugging Face models  
- Video pipeline errors on corrupt frames  
- Unstable single-model predictions  
- False positives on non-human images  
- High-dimensional vectors causing noise  

## 10.2 Fixes
- Manual label mapping  
- Switched to stable deepfake model  
- Built ensemble of three models  
- Added frame safety checks  
- Introduced threshold logic  
- Switched to 2-dimensional vectors  

---

# 11. Known Issues
- Very low-quality media may reduce accuracy  
- LLM explanations may vary  
- Browser may lag while loading large videos  
- Placeholder generator is basic  
- 2D vector limits deeper similarity patterns  

---

# 12. Contribution Guidelines
- Keep updates small and clear  
- Follow folder structure  
- Do not commit API keys  
- Attach logs for bug reports  
- Keep UI consistent with existing design  

---

# 13. Bug Reporting
1. Create an issue on GitHub  
2. Describe steps to reproduce  
3. Add expected/actual results  
4. Include logs or screenshots  
5. Submit a pull request after fixing  

---

# 14. Support and Donations
Future improvements may need better models, datasets, or compute.  
Support or contributions can be added here later.

---


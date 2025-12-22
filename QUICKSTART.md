# Quick Start Guide - SHJ Category Learning Experiment

## For Participants (You & Your Friends)

### Running the Experiment

**METHOD 1: Using the Local Server (Recommended)**

1. **Start the server** (one-time setup):
   - Open a terminal/command prompt
   - Navigate to the project2 folder
   - Run: `python serve.py`
   - Your browser will automatically open to: http://localhost:8000

2. **Complete the task**:
   - Navigate to the `experiment/` folder
   - Double-click `index.html` to open in your web browser
   - (Or right-click → Open with → Choose your browser)

2. **Complete the task**:
   - Enter your name or initials as participant ID
   - Read the instructions carefully
   - Complete the practice trials (8 trials)
   - Complete the experimental blocks (you'll do 2-3 different category types)
   - Each block stops when you reach 16 out of 20 correct, or after 160 trials max

3. **Download your data**:
   - When you complete a block, click "Download Data (JSON)"
   - Save the file with a descriptive name
   - **Important**: Send this file back so it can be included in the analysis!

4. **Continue or finish**:
   - You can do multiple blocks in one session
   - Or take a break between blocks if needed

### Tips for Best Data Quality

- **Focus**: Minimize distractions during the task
- **Honesty**: Guess if you're unsure - this is learning data, not a test!
- **Consistency**: Try to complete at least 2 blocks if possible
- **Timing**: Each block typically takes 5-15 minutes depending on learning speed

---

## For the Researcher (You)

### Running the Neural Models

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models**:
   ```bash
   cd models
   python train.py
   ```

   This will:
   - Train logistic regression and MLP on all SHJ types
   - Run 10 replications per model per type
   - Save results to `../results/model_results.json`
   - Takes about 1-2 minutes total

3. **Check individual model files**:
   ```bash
   # Just logistic regression
   python logistic.py
   
   # Just MLP
   python mlp.py
   ```

### Analyzing the Results

1. **Organize human data**:
   - Collect all JSON files from participants
   - Place them in `experiment/data/` directory

2. **Run analysis notebooks**:
   ```bash
   cd analysis
   jupyter notebook
   ```
   
   - Open `learning_curves.ipynb`
   - Run all cells to generate comparison plots
   - Figures will be saved to `results/figures/`

3. **Create additional analyses**:
   - `difficulty_ordering.ipynb` - Compare SHJ type rankings
   - `error_analysis.ipynb` - Confusion matrices
   - `generalization.ipynb` - Train/test splits

### Sharing the Experiment with Friends

**Option 1: Local Files**
- Zip the entire `experiment/` folder
- Send to friends
- They unzip and open `index.html`

**Option 2: Host Online** (if you want to make it easier)
- Use GitHub Pages, Netlify, or similar free hosting
- Share the URL
- Data files are downloaded automatically

### Project Completion Checklist

- [ ] Collect data from at least 3-5 participants (including yourself)
- [ ] Run model training (`python models/train.py`)
- [ ] Generate learning curve plots
- [ ] Complete difficulty ordering analysis
- [ ] Document findings in the README
- [ ] (Optional) Create a short report PDF

### Expected Timeline

- **Experiment setup**: ✅ Complete!
- **Data collection**: 1-2 weeks (depending on friend availability)
- **Model training**: 5 minutes
- **Analysis & visualization**: 1-2 hours
- **Write-up**: 2-3 hours

### Troubleshooting

**Experiment won't open?**
- Make sure `stimuli.json` and `shj_mappings.json` are in the same folder as `index.html`
- Try a different browser (Chrome, Firefox, Edge all work)

**Models fail to train?**
- Check Python version (3.7+ required)
- Verify numpy is installed: `pip install numpy`
- Make sure you're in the `models/` directory

**Analysis notebook errors?**
- Install jupyter: `pip install jupyter`
- Check that human data files are in `experiment/data/`

### Questions?

This is a cognitive science research project comparing human and neural network learning. The README.md has full technical details if you want to dive deeper!

---

**Good luck with data collection! 🧠🤖**
